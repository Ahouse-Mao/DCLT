import pytorch_lightning as L
import os
import numpy as np
import torch
import torch.nn as nn
from einops import reduce

from cl_models.ChunkingPatch import ChunkingPatch
from loss.infonce_loss import InfoNCE

from data_provider.DCLT_data_loader_v2 import GraphContrastDataset

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

class LitModel(L.LightningModule):
    def __init__(self, cfg, T):
        super().__init__()
        self.cfg = cfg
        self.lr = cfg.light_model.optimizer.lr
        self.model = self.select_model()
        self.head_args = self.cfg.light_model.head
        self.T = T
        self._cal_n_chunks()
        self._init_head()
        self.loss = InfoNCE(temperature=0.1, reduction='mean', negative_mode='paired')
        self.save_hyperparameters()

    def select_model(self):
        if self.cfg.model_name == 'DCLT_patchtst_pretrained_cl':
            return ChunkingPatch(cfg=self.cfg)

    def configure_optimizers(self):
        """配置优化器(AdamW, 可按需扩展 scheduler)"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def _cal_n_chunks(self):
        """用于计算n_chunks的(n_chunks会在proj_head的初始化中用到)"""
        T = self.T
        chunk_len = self.cfg.light_model.chunking.chunk_len
        overlap = self.cfg.light_model.chunking.overlap
        stride  = int(chunk_len * (1 - float(overlap)))
        self.n_chunks = (T - chunk_len + stride - 1) // stride + 1 # 因为内部的n_chunks计算中，使用了最后一段不足则pad的操作，所以这里修改了一下公式
        # 内部逻辑等价于: if T <= L: n_chunks = 1 else:
        # # 若 remainder>0 先 pad stride - remainder T_pad = T if remainder==0 else T + (stride - remainder) n_chunks = (T_pad - L)//stride + 1
        # 化简成统一公式(“ceil”形式): T <= L → n_chunks = 1 T > L → n_chunks = ceil((T - L)/stride) + 1

    def _init_head(self):
        """初始化proj_head"""
        self.proj_1 = nn.Sequential(
            nn.Linear(self.model.backbone.out_dim, self.head_args.hidden_dim_1, self.head_args.bias),
            nn.ReLU(inplace=True),
            nn.Dropout(self.head_args.dropout),
            nn.Linear(self.head_args.hidden_dim_1, self.head_args.out_dim_1)
        )

        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1) # 把 n_chunks 和 out_dim_1 维度展平

        self.proj_2 = nn.Sequential(
            nn.Linear(self.head_args.out_dim_1 * self.n_chunks, self.head_args.hidden_dim_2, self.head_args.bias),
            nn.ReLU(inplace=True),
            nn.Dropout(self.head_args.dropout),
            nn.Linear(self.head_args.hidden_dim_2, self.head_args.final_out_dim)
        )

    def proj_head(self, x):
                """proj_head的前向传播
                支持输入形状:
                    - (B, n_chunks, D)            (会自动视为 S=1)
                    - (B, S, n_chunks, D)
                输出: (B, S, D_out)
                """
                if x.dim() == 3:  # (B, n_chunks, D)
                        x = x.unsqueeze(1)  # (B, 1, n_chunks, D)
                B, S, N, D = x.shape
                x = x.reshape(B * S * N, D)             # (B*S*n_chunks, D)
                x = self.proj_1(x)                      # (B*S*n_chunks, D')
                x = x.reshape(B * S, N, -1)             # (B*S, n_chunks, D')
                x = self.flatten(x)                     # (B*S, n_chunks*D')
                x = self.proj_2(x)                      # (B*S, D_out)
                x = x.reshape(B, S, -1)                 # (B, S, D_out)
                return x

    def _init_loss(self):
        """初始化loss函数"""
        # 一个loss是sim_loss，另外一个是传统的对比学习使用的InfoNCE_loss
        # sim_loss用于教会模型区分相似与不相似
        tempreture = self.cfg.light_model.loss.temperature
        reduction = self.cfg.light_model.loss.reduction
        negative_mode = self.cfg.light_model.loss.negative_mode
        self.loss = InfoNCE(temperature=tempreture, reduction=reduction, negative_mode=negative_mode)

    def loss_cal(self, z_a_head, z_p_head, z_n_head):
        """
        计算一个 batch 的 InfoNCE。
        输入:
          z_a_head: (B, 1, D)
          z_p_head: (B, P, D)
          z_n_head: (B, N, D)
        逻辑:
          - 对于每个样本 i（batch 中的第 i 个 anchor），复制该 anchor 向量 P 次，
            按 (i, j) 与对应的第 j 个 positive 配对；
          - paired negatives 取该样本自己的 N 个 negatives，不与其它样本交叉。
        """
        B, A, D = z_a_head.shape
        P = z_p_head.shape[1]
        N = z_n_head.shape[1]

        # (B, 1, D) -> (B, P, D) -> (B*P, D)
        query = z_a_head.expand(B, P, D).reshape(B * P, D).contiguous()

        # positives: (B, P, D) -> (B*P, D)
        pos = z_p_head.reshape(B * P, D).contiguous()

        # negatives (paired): (B, N, D) -> 复制到每个正样本 (B, P, N, D) -> (B*P, N, D)
        neg = (
            z_n_head.unsqueeze(1)                   # (B, 1, N, D)
                    .expand(B, P, N, D)             # (B, P, N, D)
                    .reshape(B * P, N, D)           # (B*P, N, D)
                    .contiguous()
        )

        # 计算 InfoNCE
        loss = self.loss(query, pos, neg)
        return loss

    def training_step(self, batch, batch_idx):
        """
        训练的时候, 训练TST模型的表征能力, 使得模型能尽可能区分正负样本
        """
        anchor, pos_data, neg_data = batch
        # anchor: (B, 1, T)
        # pos_data: (B, P, T)
        # neg_data: (B, N, T)
        B = anchor.size(0)
        P = pos_data.size(1)
        N = neg_data.size(1)

        # 编码得到嵌入,
        # encoder: (batch, T) -> (batch, D)
        data = self.model(anchor, pos_data, neg_data)

        # 分离anchor, pos, neg
        z_a = data['anchor']  # (B, n_chunks, out_dim)
        z_p = data['pos']     # (B, P, n_chunks, out_dim)
        z_n = data['neg']     # (B, N, n_chunks, out_dim)

        # head层
        z_a_head = self.proj_head(z_a)          # -> (B,1,D)
        z_p_head = self.proj_head(z_p)          # -> (B,P,D)
        z_n_head = self.proj_head(z_n)          # -> (B,N,D)

        # loss计算
        loss = self.loss_cal(z_a_head, z_p_head, z_n_head)

        # 记录训练损失
        if hasattr(self, 'log'):
            self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=B)
            self.log('train/loss', loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=B)

        return loss

    # def validation_step(self, batch, batch_idx):
    #     anchor, pos, neg = batch

    #     B = anchor.size(0)
    #     P = pos.size(1)
    #     N = neg.size(1)

    #     # TODO：这里要送入模型得到输出
    #     z_a = self.model(anchor)

    #     pos_flat = pos.reshape(B * P, pos.size(-1))
    #     z_p_flat = self.model(pos_flat)  # (B*P, D)

    #     neg_flat = neg.reshape(B * N, neg.size(-1))
    #     z_n_flat = self.model(neg_flat)  # (B*N, D)
    #     D = z_a.size(-1)
    #     z_n = z_n_flat.view(B, N, D)  # (B, N, D)

    #     # TODO：计算损失
    #     val_loss = 0

    #     # logging
    #     self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)

    #     return val_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # 预测阶段：返回 encoder embedding h（用于下游或检索）
        x = batch if not isinstance(batch, (list, tuple)) else batch[0]
        with torch.no_grad():
            h, z = self.forward(x)
        return {"h": h.detach().cpu(), "z": z.detach().cpu()}
    
    def forward(self, x):
        # x: (Batch, C_in, T_length)
        # 前向传播接口：返回 encoder embedding h 和 proj_head 输出 z
        
        h_dict = self.model(x) 
        h = h_dict['anchor'] # (B, C_in, n_chunks, out_dim)
        z = self.proj_head(h) # (B, C_in, final_out_dim)
        return h, z



# @hydra.main(version_base=None, config_path="cl_conf", config_name="pretrain_cfg")
# def main(cfg: DictConfig) -> None:
#     dataset = GraphContrastDataset(data_path=cfg.dataset.path, dtw_path=cfg.dataset.dtw_path)

#     train_loader = DataLoader(dataset, batch_size=cfg.light_model.trainer.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
#     # val_loader = DataLoader(dataset, batch_size=cfg.light_model.trainer.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
#     T = dataset.data_length
#     model = LitModel(cfg, T)

#     trainer = L.Trainer(max_epochs=cfg.light_model.trainer.max_epochs, accelerator="auto", devices=1, log_every_n_steps=1)
#     trainer.fit(model, train_loader,val_dataloaders= None)

if __name__ == "__main__":
    from data_provider.DCLT_data_loader_v2 import GraphContrastDataset
    from torch.utils.data import DataLoader

    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    from utils.Mypydebug import show_shape
    show_shape()

    with initialize_config_dir(config_dir='/home/wms/South/MyModel/cl_conf', version_base=None):
        cfg = compose(config_name='pretrain_cfg')

    chunk_model = ChunkingPatch(cfg)

    dataset = GraphContrastDataset(
        data_path="./Dataset/electricity.csv",
        dtw_matrix="./dtw_results/electricity_dtw_analysis.csv",
        k=10,
        P=5,
        N=20,
        sigma_method='self_tuning',
        self_tuning_k=10,
        use_mutual=True,
        neg_sampling='hard'
    )

    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=dataset.collate_fn)

    T = dataset.data_length
    model = LitModel(cfg, T)

    trainer = L.Trainer(max_epochs=cfg.light_model.trainer.max_epochs, accelerator="auto", devices=1, log_every_n_steps=1)
    trainer.fit(model, train_loader,val_dataloaders= None)