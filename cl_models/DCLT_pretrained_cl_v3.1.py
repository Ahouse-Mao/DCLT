import pytorch_lightning as L
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import rotary_embedding_torch as rotary
import torch.nn.functional as F

# 确保项目根目录在 sys.path 中，以便 'layers' 等顶层模块可被导入
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.Mypydebug import show_shape
show_shape()

from layers.RevIN import RevIN
from omegaconf import OmegaConf, DictConfig
from layers.CL_layers_v3 import LearnableDecompose, Encoder, FusionModule, TokenProjectionHead, TokenDecoder, Soft_CL_weight
from layers.CL_layers_v3 import spectral_loss_from_raw, token_ortho_loss

from loss.infonce_loss import InfoNCE

from data_provider.DCLT_data_loader_v3 import DCLT_data_loader_v3
from torch.utils.data import DataLoader

EPS = 1e-12
NEG_INF = -1e9


class LitModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        # 训练参数
        self.cl_shift_type = cfg.model.cl.shift_type  # 'shift_trend', 'shift_season', 'shift_both'
        self.lr = cfg.train.lr

        # loss参数
        self.loss_chunks = cfg.model.cl_loss.loss_chunks
        self.cl_weight = cfg.model.cl_loss.cl_weight
        self.spectral_weight = cfg.model.cl_loss.spectral_weight
        self.ortho_weight = cfg.model.cl_loss.ortho_weight
        self.rec_weight = cfg.model.cl_loss.rec_weight

        self.w_enh = cfg.model.cl_loss.w_enh
        self.w_neighbor = cfg.model.cl_loss.w_neighbor
        self.temp = cfg.model.cl_loss.temperature

        # revin参数
        self.revin = cfg.model.revin.use_revin
        self.n_vars = cfg.dataset.n_vars
        if self.revin:
            self.revin_layer = RevIN(
                self.n_vars,
                affine=cfg.model.revin.affine,
                subtract_last=cfg.model.revin.subtract_last,
            )

        # patch的参数
        self.seq_len = cfg.model.seq_len
        self.patch_len = cfg.model.patch_len
        self.stride = cfg.model.stride
        self.padding_patch = cfg.model.padding_patch

        patch_num = int((self.seq_len - self.patch_len) / self.stride + 1)
        if self.padding_patch == "end":  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
            patch_num += 1

        # decomposition
        self.decompose_layer = LearnableDecompose(
            cfg.dataset.n_vars, kernel_size=cfg.model.decompose.kernel_size
        )

        # encoder
        # out_dim因为维度对不齐的问题暂时不考虑
        self.trend_encoder = Encoder(
            d_model=self.patch_len,
            n_heads=cfg.model.encoder.n_heads,
            out_dim=cfg.model.encoder.out_dim,
            dropout=cfg.model.encoder.dropout,
            d_ff=cfg.model.encoder.d_ff,
            activation="gelu",
        )
        self.season_encoder = Encoder(
            d_model=self.patch_len,
            n_heads=cfg.model.encoder.n_heads,
            out_dim=cfg.model.encoder.out_dim,
            dropout=cfg.model.encoder.dropout,
            d_ff=cfg.model.encoder.d_ff,
            activation="gelu",
        )

        # fusion module
        self.fusion = FusionModule(
            d_model=self.patch_len, fusion_type=cfg.model.fusion.fusion_type
        )

        # projection head
        self.proj_head = TokenProjectionHead(
            in_dim=self.patch_len,
            hidden_dim=3 * self.patch_len,
            proj_dim=2 * self.patch_len,
        )

        # reconstruction module
        self.decoder = TokenDecoder(
            proj_dim=2 * self.patch_len,
            num_patches=patch_num,
            seq_len=self.seq_len,
            out_channels=self.n_vars,
            weight_mse=cfg.model.decoder.weight_mse,
            weight_cos=cfg.model.decoder.weight_cos,
        )

        # 对比学习权重计算类初始化
        self.soft_cl_weight = Soft_CL_weight(
            sigma=cfg.model.cl_loss.sigma,
            invert=cfg.model.cl_loss.invert,
            min_weight=cfg.model.cl_loss.min_weight,
            normalize=cfg.model.cl_loss.normalize,
        )

    def configure_optimizers(self):
        """配置优化器(AdamW, 可按需扩展 scheduler)"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        # norm
        x_remain, y = batch
        x = x_remain

        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, "norm")
            x = x.permute(0, 2, 1)

        # do patching
        if self.padding_patch == "end":
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.permute(0, 1, 3, 2)
        B, C, P, N = x.shape
        x = x.reshape(B * N, C, P)

        # decompose
        x_trend, x_season = self.decompose_layer(x)

        x_trend = x_trend.reshape(B, N, C, P)
        x_season = x_season.reshape(B, N, C, P)
        x_trend = x_trend.reshape(B * C, N, P)
        x_season = x_season.reshape(B * C, N, P)

        # encoder
        x_trend_emb = self.trend_encoder(x_trend)
        x_season_emb = self.season_encoder(x_season)

        # generate samples by using fusion module
        self_samples, enhance_samples = self.gennerate_cl_samples(
            x_trend_emb, x_season_emb
        )

        # projection head
        x_proj = self.proj_head(self_samples)
        x_proj_enhance = self.proj_head(enhance_samples)

        # reconstruction
        x_rec = self.decoder(x_proj)

        other_loss = self.other_loss_calculator(
            x_trend, x_season, x_trend_emb, x_season_emb, x_rec, x_remain
        )
        x_proj = x_proj.reshape(B, C, N, 1, -1)
        K = x_proj_enhance.shape[2]
        x_proj_enhance = x_proj_enhance.reshape(B, C, N, K, -1)
        loss = self.loss_calculator(x_proj, x_proj_enhance)

        return loss + other_loss

    def gennerate_cl_samples(self, x_trend_emb, x_season_emb):
        """
        生成对比学习样本
        input:
        x_trend_emb: (B*C, N, patch_len)
        x_season_emb: (B*C, N, patch_len)
        output:
        self_samples: (B*C, N, 1 or 2, out_dim)
        enhance_samples: (B*C, N, 1 or 2, out_dim)
        """

        B_C, N, P = x_trend_emb.shape
        self_samples = self.fusion(x_trend_emb, x_season_emb)
        self_samples = self_samples.unsqueeze(2)

        if self.cl_shift_type == "shift_trend":
            x_trend_emb_enhance = self.change_order(x_trend_emb)
            enhance_samples = self.fusion(x_trend_emb_enhance, x_season_emb)
            enhance_samples = enhance_samples.unsqueeze(2)
        elif self.cl_shift_type == "shift_season":
            x_season_emb_enhance = self.change_order(x_season_emb)
            enhance_samples = self.fusion(x_trend_emb, x_season_emb_enhance)
            enhance_samples = enhance_samples.unsqueeze(2)
        elif self.cl_shift_type == "shift_both":
            x_trend_emb_enhance = self.change_order(x_trend_emb)
            x_season_emb_enhance = self.change_order(x_season_emb)
            enhance_samples_part1 = self.fusion(x_trend_emb, x_season_emb_enhance)
            enhance_samples_part2 = self.fusion(x_trend_emb_enhance, x_season_emb)
            enhance_samples_part1 = enhance_samples_part1.unsqueeze(2)
            enhance_samples_part2 = enhance_samples_part2.unsqueeze(2)
            enhance_samples = torch.cat(
                [enhance_samples_part1, enhance_samples_part2], dim=2
            )

        return self_samples, enhance_samples

    def change_order(self, x):
        """
        输入：[patch1, patch2, ..., patchN-1, patchN]
        输出：[patch2, patch3, ..., patchN, patchN-1]
        """
        x_shift = x[:, 1:, :]
        x_last_patch = x[:, -2, :].unsqueeze(1)
        out = torch.cat([x_shift, x_last_patch], dim=1)

        return out

    def other_loss_calculator(
        self, x_trend, x_season, x_trend_emb, x_season_emb, x_rec, x_remain
    ):
        # spectral loss
        spectral_loss = spectral_loss_from_raw(
            x_trend, x_season, cutoff_ratio=self.cfg.model.spectral_loss.cutoff_ratio
        )
        # token orthogonality loss
        ortho_loss = token_ortho_loss(x_trend_emb, x_season_emb)
        # rec_loss
        if self.revin:
            x_rec = x_rec.permute(0, 2, 1)
            x_rec = self.revin_layer(x_rec, "denorm")
            x_rec = x_rec.permute(0, 2, 1)

        rec_loss = self.decoder.rec_loss(x_rec, x_remain)

        loss = (
            rec_loss * self.rec_weight
            + spectral_loss * self.spectral_weight
            + ortho_loss * self.ortho_weight
        )

        return loss

    def loss_calculator(self, x_proj, x_proj_enhance, var_dtw=None):
        device = x_proj.device  # 统一获取张量所在设备，避免多次 .device 调用
        dtype = x_proj.dtype  # 记录数据类型，构造新张量时保持一致

        loss = torch.zeros(1, device=device, dtype=dtype)  # 累计 InfoNCE 损失的缓冲张量

        w_enh = self.w_enh
        w_neighbor = self.w_neighbor
        temp = self.temp

        # === 步骤概览 ===
        # 1) 在子批次内提取 anchor 向量，聚合增强样本 + 邻居得到正样本表示。
        # 2) 计算同变量与跨变量负样本的余弦相似度，并应用对应权重/掩码。
        # 3) 将正负样本的相似度拼成 logits，并加上权重的对数偏置。
        # 4) 经过 log_softmax 得到 InfoNCE 风格的损失，并按有效 anchor 平均。

        # --- STEP 1: 构造 anchor 与正样本 ---
        # x_anchor: (B, C, N, 1, d)；x_enh: (B, C, N, K, d)
        x_anchor = x_proj
        x_enh = x_proj_enhance

        B, C, N, _, d = x_anchor.shape  # B: chunk 大小, C: 变量数, N: patch 数, d: 投影维
        K_enh = x_enh.shape[3]  # 增强样本个数(通常为 2)

        # anchors: (Q, d)，其中 Q = B * C * N
        anchors = x_anchor.squeeze(3).reshape(B * C * N, d)
        anchors_norm = F.normalize(anchors, dim=1)

        pos_sum = torch.zeros_like(anchors)  # (Q, d) 初始化正样本向量和(Q, d), 后面会把增强样本、邻居样本按照权重累加进来，得到所有正样本的中心向量
        pos_weight_sum = torch.zeros(
            anchors.shape[0], 1, device=device, dtype=dtype
        )  # (Q, 1) 用于记录每个 anchor 累积的正样本权重总和，方便最后做加权平均，防止除零（配合 EPS）

        if K_enh > 0:
            # enh_flat: (Q, K, d)
            enh_flat = x_enh.reshape(B * C * N, K_enh, d)
            pos_sum = pos_sum + (enh_flat * w_enh).sum(dim=1) # 增强样本加权累加
            pos_weight_sum = pos_weight_sum + w_enh * torch.ones_like(pos_weight_sum)

        # 提前计算BxCxN每个 anchor 在 batch / 变量 / patch 维度上的索引
        q_idx = torch.arange(anchors.shape[0], device=device)  # (Q,)
        n_idx = q_idx % N  # 通过取模得到每个anchor属于第几个patch
        bc_idx = q_idx // N  # 通过整除得到每个anchor属于第几个变量序列
        b_idx = bc_idx // C  # 得到批次索引, batch index (0..B-1)
        c_idx = bc_idx % C  # 得到变量索引, variable index (0..C-1)

        # valid_left/right: (Q,) 布尔向量，标记 anchor 是否存在左/右邻居 patch
        valid_left = n_idx > 0
        if valid_left.any(): # 对于存在左邻居的anchor
            left_indices = q_idx[valid_left] - 1  # 邻居 patch 的展平索引
            pos_sum[valid_left] += w_neighbor * anchors[left_indices] # 左邻居样本作为正样本加权乘上去
            pos_weight_sum[valid_left] += w_neighbor

        valid_right = n_idx < (N - 1)
        if valid_right.any(): # 对于存在右邻居的anchor
            right_indices = q_idx[valid_right] + 1  # 右邻居 patch 的展平索引
            pos_sum[valid_right] += w_neighbor * anchors[right_indices] # 右邻居样本作为正样本加权乘上去
            pos_weight_sum[valid_right] += w_neighbor # 权重累加

        valid_pos_mask = pos_weight_sum.squeeze(1) > 0  # 标记哪些 anchor 真正拥有正样本
        positive_key = pos_sum / (pos_weight_sum + EPS)  # 对正样本累积向量做加权平均，得到“正样本中心”，EPS 防止除零。

        anc_norm = anchors_norm # 确保anchor和向量中心都是
        pos_norm = F.normalize(positive_key, dim=1)
        sim_pos = (anc_norm * pos_norm).sum(dim=1)

        # --- STEP 2: 计算同变量/跨变量负样本及其权重 ---
        # 2.1计算同变量负样本的权重与掩码
        # var_dtw_norm: (C, C)
        if var_dtw is not None:
            maxv = var_dtw.max()  # 同变量 DTW 矩阵中的最大值，供归一化使用
            if maxv > 0:
                var_dtw_norm = (var_dtw / (maxv + EPS)).to(device=device, dtype=dtype)
            else:
                var_dtw_norm = torch.ones((C, C), device=device, dtype=dtype) * 0.5
        else:
            var_dtw_norm = torch.ones((C, C), device=device, dtype=dtype) * 0.5

        # bc_vectors: (B*C, N, d) —— 每个变量上所有 patch 的表示，方便按变量维度聚合patch
        bc_vectors = x_anchor.squeeze(3).reshape(B * C, N, d)
        bc_norm = F.normalize(bc_vectors, dim=2)
        # same_sim_matrix: (B*C, N, N)，同变量不同 patch 的余弦相似度
        same_sim_matrix = torch.matmul(bc_norm, bc_norm.transpose(1, 2))
        # same_sim: (Q, N)，逐 anchor 拿到同变量的所有 patch 余弦相似度
        same_sim = same_sim_matrix[bc_idx, n_idx]
        
        # 计算同变量负样本的权重与掩码
        soft_weight_raw = self.soft_cl_weight.compute(n_idx, N).to(
            device=device, dtype=dtype
        )
        mask_same = torch.ones_like(soft_weight_raw, dtype=torch.bool, device=device)
        mask_same[torch.arange(anchors.shape[0], device=device), n_idx] = False
        if N > 1:
            # 为了避免把anchor的左右邻居当作负样本，这里将邻近位置的掩码关闭
            # （这些邻居已经在正样本聚合时参与过，不需重复进入 InfoNCE 分母）。
            left_target_idx = n_idx - 1
            right_target_idx = n_idx + 1
            mask_same[valid_left, left_target_idx[valid_left]] = False
            mask_same[valid_right, right_target_idx[valid_right]] = False

        soft_weight = soft_weight_raw * mask_same.float()
        same_logits = same_sim / temp  # (Q, N)
        same_logits = same_logits.masked_fill(~mask_same, NEG_INF)

        # 2.2计算跨变量负样本的权重与掩码
        patch_vectors = (
            x_anchor.squeeze(3).permute(0, 2, 1, 3).contiguous().reshape(B * N, C, d)
        )
        # patch_vectors: (B*N, C, d)，把同一 patch 下所有变量的表示整理在一起，方便后续构造跨变量相似度
        patch_norm = F.normalize(patch_vectors, dim=2)
        # diff_sim_matrix: (B*N, C, C)，同一 patch 在不同变量之间的余弦相似度
        diff_sim_matrix = torch.matmul(patch_norm, patch_norm.transpose(1, 2))
        patch_row_idx = b_idx * N + n_idx  # 把展平后的 anchor 索引映射到 “batch*patch” 维，锁定当前 anchor 所在的 patch。
        diff_sim = diff_sim_matrix[patch_row_idx, c_idx]  # (Q, C) # 
        diff_logits = diff_sim / temp

        diff_w = var_dtw_norm.index_select(0, c_idx)  # 用归一化后的 DTW 权重来衡量变量间的关联，形状 (Q, C)。
        diff_w[torch.arange(anchors.shape[0], device=device), c_idx] = 0.0  # 自身变量不参与, 设置为0
        diff_logits = diff_logits.masked_fill(diff_w <= 0, NEG_INF)

        # --- STEP 3: 拼接正负 logits 并加入权重的对数偏置 ---
        pos_weight_vec = pos_weight_sum.squeeze(1).clamp(min=EPS)  # (Q,)
        pos_logits = sim_pos.unsqueeze(1) / temp  # (Q, 1) # 计算正样本 logits

        # logits_all / weights_all: (Q, 1 + N + C)
        logits_all = torch.cat([pos_logits, same_logits, diff_logits], dim=1)
        weights_all = torch.cat(
            [pos_weight_vec.unsqueeze(1), soft_weight, diff_w], dim=1
        )

        zero_mask = weights_all <= 0
        logits_all = logits_all + torch.log(weights_all.clamp(min=EPS)) # 把权重转成对数并加到 logits 上，相当于把权重视作 softmax 前的偏置
        logits_all = logits_all.masked_fill(zero_mask, NEG_INF)

        # --- STEP 4: InfoNCE 损失并按有效 anchor 做均值 ---
        logp = F.log_softmax(logits_all, dim=1) # 对每个 anchor 的整行 logits 做 log-softmax，得到对数概率
        loss_per_anchor = -logp[:, 0]  # 取每行第 0 列（正样本）的对数概率，取负号得到 InfoNCE 的正项损失。
        loss_per_anchor = loss_per_anchor * valid_pos_mask.float() # 对没有正样本的 anchor（比如权重总和为 0）将损失置为 0，避免干扰平均。

        valid_count = valid_pos_mask.float().sum()  # 有效 anchor 的数量
        if valid_count.item() > 0:
            cl_loss = loss_per_anchor.sum() / valid_count
        else:
            cl_loss = torch.zeros(1, device=device, dtype=dtype)

        loss = cl_loss * self.cl_weight

        return loss


if __name__ == "__main__":

    cfg = OmegaConf.load("./cl_conf/pretrain_cfg_v3.yaml")

    dataset = DCLT_data_loader_v3(cfg=cfg)

    model = LitModel(cfg)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    train_loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)

    trainer = L.Trainer(
        max_epochs=cfg.train.epochs,
        devices="auto",
        log_every_n_steps=1,
        accelerator="auto",
    )
    from pytorch_lightning.utilities.model_summary import summarize

    print(summarize(model, max_depth=2))
    trainer.fit(model, train_loader, val_dataloaders=None)
