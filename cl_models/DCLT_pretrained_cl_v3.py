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


class LitModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        # 训练参数
        self.cl_shift_type = cfg.model.cl.shift_type # 'shift_trend', 'shift_season', 'shift_both'

        # revin参数
        self.revin = cfg.model.revin.use_revin
        self.n_vars = cfg.dataset.n_vars
        if self.revin: 
            self.revin_layer = RevIN(self.n_vars, affine=cfg.model.revin.affine,
                                      subtract_last=cfg.model.revin.subtract_last)

        # patch的参数
        self.seq_len = cfg.model.seq_len
        self.patch_len = cfg.model.patch_len
        self.stride = cfg.model.stride
        self.padding_patch = cfg.model.padding_patch

        patch_num = int((self.seq_len - self.patch_len)/self.stride + 1)
        if self.padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
            patch_num += 1

        # decomposition
        self.decompose_layer = LearnableDecompose(cfg.dataset.n_vars, kernel_size=cfg.model.decompose.kernel_size)
        
        # encoder
        # out_dim因为维度对不齐的问题暂时不考虑
        self.trend_encoder = Encoder(d_model=self.patch_len, n_heads=cfg.model.encoder.n_heads,
                                      out_dim=cfg.model.encoder.out_dim, dropout=cfg.model.encoder.dropout,
                                        d_ff=cfg.model.encoder.d_ff, activation="gelu")
        self.season_encoder = Encoder(d_model=self.patch_len, n_heads=cfg.model.encoder.n_heads,
                                       out_dim=cfg.model.encoder.out_dim, dropout=cfg.model.encoder.dropout,
                                         d_ff=cfg.model.encoder.d_ff, activation="gelu")

        # fusion module
        self.fusion = FusionModule(d_model=self.patch_len, fusion_type=cfg.model.fusion.fusion_type)

        # projection head
        self.proj_head = TokenProjectionHead(in_dim=self.patch_len, hidden_dim=3*self.patch_len,
                                              proj_dim=2*self.patch_len)

        # reconstruction module
        self.decoder = TokenDecoder(proj_dim=2*self.patch_len, num_patches=patch_num,
                                     seq_len=self.seq_len, out_channels=self.n_vars,
                                       weight_mse=cfg.model.decoder.weight_mse, weight_cos=cfg.model.decoder.weight_cos)

        # 对比学习权重计算类初始化
        self.soft_cl_weight = Soft_CL_weight(sigma=cfg.model.cl_loss.sigma, invert=cfg.model.cl_loss.invert,
                                        min_weight=cfg.model.cl_loss.min_weight, normalize=cfg.model.cl_loss.normalize,
                                        )

    # def train(self, batch, batch_idx):
    #     # norm
    #     x_remain, y = batch
    #     x = x_remain


    def train(self, x, y):
        # norm
        x_remain =x
        
        if self.revin: 
            x = x.permute(0,2,1)
            x = self.revin_layer(x, 'norm')
            x = x.permute(0,2,1)
        
        # do patching
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # x: [bs x nvars x patch_num x patch_len]
        x = x.permute(0,1,3,2)
        B, C, P, N = x.shape
        x = x.reshape(B*N, C, P)
        
        # decompose
        x_trend, x_season = self.decompose_layer(x) # x_trend, x_season: (B*N, C, P)

        # x_trend, x_season去计算频谱损失spectral_loss

        x_trend= x_trend.reshape(B, N, C, P)
        x_season= x_season.reshape(B, N, C, P)
        x_trend = x_trend.reshape(B*C, N, P)
        x_season = x_season.reshape(B*C, N, P)

        # encoder
        x_trend_emb = self.trend_encoder(x_trend)  # x_emb: (B*C, N, patch_len) -> y: (B*C, N, out_dim)
        x_season_emb = self.season_encoder(x_season)  # x_emb: (B*C , N, patch_len) -> y: (B*C, N, out_dim)

        # x_trend_emb, x_season_emb去2计算正交损失token orthogonality loss

        # generate samples by using fusion module
        self_samples, enhance_samples = self.gennerate_cl_samples(x_trend_emb, x_season_emb)

        # projection head
        x_proj = self.proj_head(self_samples) # (B*C, N, 1, proj_dim) proj的是高维表示
        x_proj_enhance = self.proj_head(enhance_samples) # (B*C, N, K, proj_dim)

        # reconstruction
        x_rec = self.decoder(x_proj) # (B, C, L)
        # x_rec and x_remain 去计算重建损失rec_loss
        
        loss = self.loss_calculator(x_trend, x_season, x_trend_emb, x_season_emb, x_rec, x_remain, x_proj, x_proj_enhance)

    def gennerate_cl_samples(self, x_trend_emb, x_season_emb):
        """
        生成对比学习样本
        input:
        x_trend_emb: (B*C, N, patch_len)
        x_season_emb: (B*C, N, patch_len)
        output:
        self_samples: (B*C, N, 1 or 2, out_dim) 最后一维的形状变化在fusion模块里变的
        enhance_samples: (B*C, N, 1 or 2, out_dim)
        """

        B_C, N, P = x_trend_emb.shape
        self_samples = self.fusion(x_trend_emb, x_season_emb) # (B*C, N, out_dim)
        self_samples = self_samples.unsqueeze(2) # (B*C, N, 1, out_dim)

        if self.cl_shift_type == 'shift_trend':
            x_trend_emb_enhance = self.change_order(x_trend_emb) # (B*C, N, patch_len)
            enhance_samples = self.fusion(x_trend_emb_enhance, x_season_emb) # (B*C, N, out_dim)
            enhance_samples = enhance_samples.unsqueeze(2) # (B*C, N, 1, out_dim)
        elif self.cl_shift_type == 'shift_season':
            x_season_emb_enhance = self.change_order(x_season_emb) # (B*C, N, patch_len)
            enhance_samples = self.fusion(x_trend_emb, x_season_emb_enhance) # (B*C, N, out_dim)
            enhance_samples = enhance_samples.unsqueeze(2) # (B*C, N, 1, out_dim)
        elif self.cl_shift_type == 'shift_both':
            x_trend_emb_enhance = self.change_order(x_trend_emb) # (B*C, N, patch_len)
            x_season_emb_enhance = self.change_order(x_season_emb) # (B*C, N, patch_len)
            enhance_samples_part1 = self.fusion(x_trend_emb, x_season_emb_enhance) # (B*C, N, out_dim)
            enhance_samples_part2 = self.fusion(x_trend_emb_enhance, x_season_emb) # (B*C, N, out_dim)
            enhance_samples_part1 = enhance_samples_part1.unsqueeze(2) # (B*C, N, 1, out_dim)
            enhance_samples_part2 = enhance_samples_part2.unsqueeze(2) # (B*C, N, 1, out_dim)
            enhance_samples = torch.cat([enhance_samples_part1, enhance_samples_part2], dim=2) # (B*C, N, 2, out_dim)

        return self_samples, enhance_samples
    
    def change_order(self, x):
        """ 
        输入：[patch1, patch2, ..., patchN-1, patchN]
        输出：[patch2, patch3, ..., patchN, patchN-1]
        """
        x_shift = x[:, 1:, :] # [patch2, patch3, ..., patchN],[Bp, N-1, P]
        x_last_patch = x[:, -2, :].unsqueeze(1) # [patchN-1],[Bp, 1, P]
        out = torch.cat([x_shift, x_last_patch], dim=1)

        return out

    def loss_calculator(self, x_trend, x_season, x_trend_emb, x_season_emb, x_rec, x_remain, x_proj, x_proj_enhance, var_dtw=None):
        # spectral loss
        spectral_loss = spectral_loss_from_raw(x_trend, x_season, cutoff_ratio=self.cfg.model.spectral_loss.cutoff_ratio)
        # token orthogonality loss
        ortho_loss = token_ortho_loss(x_trend_emb, x_season_emb)
        # rec_loss
        if self.revin: # denorm
            x_rec = x_rec.permute(0,2,1)
            x_rec = self.revin_layer(x_rec, 'denorm')
            x_rec = x_rec.permute(0,2,1)

        rec_loss = self.decoder.rec_loss(x_rec, x_remain)

        # contrastive loss
        B = x_remain.shape[0]
        C = self.n_vars

        # x_proj: (B*C, N, 1, d)
        # x_proj_enhance: (B*C, N, K, d)

        BC, N, K1, d = x_proj.shape
        K_enh = x_proj_enhance.shape[2]

        # reshape
        x_proj = x_proj.reshape(B, C, N, K1, d).squeeze(3) # (B, C, N, d)
        x_proj_enhance = x_proj_enhance.reshape(B, C, N, K_enh, d) # (B, C, N, K, d)

        # hyperparams
        cl_cfg = getattr(self.cfg.model, "cl", None) 
        w_enh = float(getattr(cl_cfg, "w_enhance", 2.0)) if cl_cfg is not None else 2.0 # 增强样本正样本权重 (大一些)
        w_neighbor = float(getattr(cl_cfg, "w_neighbor", 0.8)) if cl_cfg is not None else 0.8 # 左右邻居正样本权重
        gamma = float(getattr(cl_cfg, "same_var_gamma", 1.0)) if cl_cfg is not None else 1.0 # 同变量负样本权重的指数
        temp = float(getattr(cl_cfg, "temperature", 0.1)) if cl_cfg is not None else 0.1 # 温度系数
        # optinal DTW matrix
        # 这里从外部获取dtw, 有利于后续扩展到计算不同batch的dtw, 在epoch1时完成后续所需的dtw计算
        # 外源信息的格式应当是(n_vars, n_vars)的矩阵
        if var_dtw is not None:
            maxv = var_dtw.max()
            if maxv > 0:
                var_dtw_norm = var_dtw / (maxv + 1e-12)
            else:
                var_dtw_norm = torch.ones_like(var_dtw) * 0.5
        else:
            var_dtw_norm = torch.ones((C, C)) * 0.5


        # 准备各类样本
        Q = B * C * N
        anchors = x_proj.reshape(Q, d) # (Q, d) anchors
        enh_flat = x_proj_enhance.reshape(Q, K_enh, d) # (Q, K, d) 增强样本正样本

        # 构造左右邻居 (per (B*C, N, d))
        x_proj_bc_n = x_proj.reshape(B * C, N, d)   # (BC, N, d)
        # 左邻居
        left = torch.zeros_like(x_proj_bc_n) # left: (BC, N, d)
        left[:, 1:, :] = x_proj_bc_n[:, :-1, :]
        # 把x_proj_bc_n从index=0开始的值赋给left从index=1开始到index=最后一个的值
        # 第一个没有左邻居，所以保持为0
        # 右邻居
        right = torch.zeros_like(x_proj_bc_n)  # right: (BC, N, d)
        right[:, :-1, :] = x_proj_bc_n[:, 1:, :] 
        # 把x_proj_bc_n从index=1开始的值赋给right从index=0开始到index=-1（倒数第二个）的值
        # 最后一个没有右邻居，所以保持为0
        left_flat = left.reshape(Q, d) # (Q, d) 把left展平
        right_flat = right.reshape(Q, d) # (Q, d) 把right展平

        # 同一变量的所有patch集合，排除自身的mask放到后续
        same_var_all = x_proj_bc_n.unsqueeze(1).repeat(1, N, 1, 1) # (BC, N, N, d)
        same_var_all_flat = same_var_all.reshape(Q, N, d) # (Q, N, d)

        # 不同变量的所有patch集合，排除自身的mask放到后续
        x_proj_per_patch = x_proj.permute(0, 2, 1, 3) # x_proj: (B, C, N, d) -> (B, N, C, d)
        diff_var_all = x_proj_per_patch.unsqueeze(1).repeat(1, C, 1, 1, 1) # (B, C, N, C, d)
        diff_var_all_flat = diff_var_all.reshape(Q, C, d) # (Q, C, d)

        # 构造 candidate 矩阵
        # 每个anchor的候选集合[增强样本K, 左邻居, 右邻居, 同变量其他patch(N), 不同变量所有patch(C)]

        M = K_enh + 1 + 1 + N + C
        cand = torch.zeros(Q, M, d) # (Q, M, d)

        pos = 0
        cand[:, pos:pos+K_enh, :] = enh_flat # 增强样本正样本 (Q, K, d)
        pos += K_enh
        cand[:, pos, :] = left_flat # 左邻居正样本 (Q, 1, d)
        pos += 1
        cand[:, pos, :] = right_flat # 右邻居正样本 (Q, 1, d)
        pos += 1
        cand[:, pos:pos+N, :] = same_var_all_flat # 同变量负样本 (Q, N, d)
        pos += N
        cand[:, pos:pos+C, :] = diff_var_all_flat # 不同变量负样本 (Q, C, d)
        pos += C

        # 构造权重矩阵
        weights = torch.zeros(Q, M)# (Q, M)
        pos = 0

        weights[:, pos:pos+K_enh] = w_enh # 增强样本正样本权重 (Q, K)
        pos += K_enh

        # 左右邻居权重, 只有在存在邻居的情况下才有权重
        anchor_idx = torch.arange(Q) # (Q,)
        n_idx = anchor_idx % N  # (Q,)
        left_exist = (n_idx > 0).float() # (Q,) 选出有左邻居的，忽略起始点0
        right_exist = (n_idx < (N - 1)).float() # (Q,) 选出有右邻居的，忽略最后一个点
        weights[:, pos] = w_neighbor * left_exist  # 左邻居正样本权重 (Q, 1)
        pos += 1
        weights[:, pos] = w_neighbor * right_exist  # 右邻居正样本权重 (Q, 1)
        pos += 1

        # 软对比学习权重(未mask)
        soft_weight_raw = self.soft_cl_weight.compute(n_idx, N) # (Q, N)

        # 计算mask
        m_idx = torch.arange(N).unsqueeze(0)  # (1, N) m_idx代表候选集中的同变量patch的index
        n_idx_repeat = (n_idx.unsqueeze(1)).float()  # (Q, 1)
        dist = torch.abs(n_idx_repeat - m_idx).float()  # (Q, N) 每个anchor到同变量所有patch的距离, Q=B*C*N, dist也是arrange产生的
        mask_same = ((m_idx != n_idx_repeat) &
                     (m_idx != (n_idx_repeat - 1)) &
                     (m_idx != (n_idx_repeat + 1))).float()  # (Q, N) 排除自身和左右邻居
        # 应用mask
        soft_weight = soft_weight_raw * mask_same # (Q, N) 只保留同变量负样本的权重，其他位置为0

        # 写回weights
        weights[:, pos:pos+N] = soft_weight # 同变量负样本权重 (Q, N)
        pos += N

        # 不同变量相同位置的负样本权重, 来自var_dtw_norm
        c_base = torch.arange(C).unsqueeze(1).repeat(1, N).reshape(-1) # (C*N,)
        c_idx = c_base.repeat(B) # (Q,)=(B*C*N, ) 每个anchor对应的变量index
        # 用高级索引：取出对应行 -> diff_w shape (Q, C), 因为Q=B*C*N，把C提取出来
        diff_w = var_dtw_norm[c_idx]
        diff_w[torch.arange(Q), c_idx] = 0.0 # 排除自身
        weights[:, pos:pos + C] = diff_w # 写入weights
        pos += C

        # 构造正样本向量
        pos_pos_end = K_enh + 1 + 1 # 正样本在候选集中的结束位置
        pos_weights = weights[:, :pos_pos_end] # (Q, K+1+1)
        pos_vectors = cand[:, :pos_pos_end, :] # (Q, K+1+1, d)
        # 每个 anchor 在正样本段上的总权重（标量）。形状 (Q,1)。
        pos_weight_sum = pos_weights.sum(dim=1, keepdim=True) # (Q, 1)
        # 布尔掩码，表示哪些 anchor 至少有一个正样本（权重和 > 0）。后续会跳过没有正样本的 anchor（防止除零或产生无意义的 loss）。
        valid_pos_mask = (pos_weight_sum.squeeze(1) > 0).float() # (Q,) 选出有正样本的anchor
        
        # 按权重合成正样本向量
        positive_key = torch.sum(pos_vectors * pos_weights.unsqueeze(2), dim=1) # (Q, d)
        positive_key = positive_key / (pos_weight_sum + 1e-12) # (Q, d) 避免除零

        # 构造负样本向量
        neg_weights = weights[:, pos_pos_end:] # (Q, N+C)
        neg_vectors = cand[:, pos_pos_end:, :] # (Q, N+C, d)

        # 下面是计算InfoNCE loss
        # 归一化向量，计算余弦相似度
        # 首先对所有向量进行L2归一化
        anc_n = anchors / (anchors.norm(dim=1, keepdim=True) + 1e-12) # (Q, d)
        pos_n = positive_key / (positive_key.norm(dim=1, keepdim=True) + 1e-12) # (Q, d)
        neg_n = neg_vectors / (neg_vectors.norm(dim=2, keepdim=True) + 1e-12) # (Q, N+C, d)

        sim_pos = (anc_n * pos_n).sum(dim=1)  # (Q,) 余弦相似度
        sim_neg = torch.bmm(neg_n, anc_n.unsqueeze(2)).squeeze(2)  # (Q, N+C)

        # 构造logits, 并做temperature缩放
        logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1) # (Q, 1+N+C)
        logits = logits / float(temp) # 温度系数加入，计算出logits

        # 构造权重行并加入logits
        pos_weight_vec = pos_weight_sum.squeeze(1).clamp(min=1e-12) # (Q,) 避免除零
        w_row = torch.cat([pos_weight_vec.unsqueeze(1), neg_weights], dim=1) # (Q, 1+N+C)

        # 对0权重的样本做mask
        zero_mask = (w_row == 0.0) # (Q, 1+N+C)
        tiny = 1e-12
        w_row_clamped = w_row.clamp(min=tiny)                  # 将 0 -> tiny 以避免 log(0)
        logits = logits + torch.log(w_row_clamped)
        logits = logits.masked_fill(zero_mask, -1e9)

        logp = F.log_softmax(logits, dim=1)
        loss_per_anchor = -logp[:, 0]    # (Q,)

        if (~valid_pos_mask).any():
            loss_per_anchor = loss_per_anchor * valid_pos_mask.float()

        valid_count = valid_pos_mask.float().sum()
        if valid_count.item() == 0:
            cl_loss = torch.tensor(0.0)
        else:
            cl_loss = loss_per_anchor.sum() / valid_count

        # 总损失
        loss = cl_loss * self.cfg.model.cl_loss.weight + \
               rec_loss * self.cfg.model.rec_loss.weight + \
               spectral_loss * self.cfg.model.spectral_loss.weight + \
               ortho_loss * self.cfg.model.ortho_loss.weight

        return loss

if __name__ == "__main__":
    cfg = OmegaConf.load('./cl_conf/pretrain_cfg_v3.yaml')
    model = LitModel(cfg)
    x = torch.rand(16, cfg.dataset.n_vars, cfg.model.seq_len)
    y = torch.rand(16, cfg.dataset.n_vars, cfg.model.pred_len)
    model.train(x, y)


        