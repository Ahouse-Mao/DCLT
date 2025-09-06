__all__ = ['PatchTST_feat_backbone']

"""
PatchTST module with Feature-Extraction wrapper

说明：
- 类 PatchTSTFeatureBackbone：用于把 patchtst 输出转为固定维度 embedding，适合作为 chunk-level encoder（在注释中标为 MODIFIED/ADDED）。
- 所有新增/修改位置在注释中清楚标注为 "=== MODIFIED/ADDED ==="。
- 所有原始未改动的代码块在注释中标注为 "=== UNCHANGED ==="。
"""

# Cell
from typing import Callable, Literal, Optional

# test cell
import os, sys
# 允许直接运行该文件时自动把项目根目录加入 sys.path，避免手动 export PYTHONPATH
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
# test cell

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.RevIN import RevIN
from layers.PatchTST_backbone import TSTiEncoder

# ============================
# === MODIFIED/ADDED: PatchTSTFeatureBackbone （新增） ===
# 说明：下面的类是我新增的 wrapper，用来把 patchtst 的 patch-level输出聚合成固定维度 embedding，
#       适合作为 ChunkingEncoderV2 中对每个 chunk 的 encoder。
# 修改要点都在注释里标明（"=== MODIFIED/ADDED ==="），并用中文解释每个参数与实现细节。
# ============================

class PatchTST_feat_backbone(nn.Module):
    """
    === MODIFIED/ADDED ===
    PatchTST 特征提取器 wrapper（新增类）
    
    功能：
      - 保留原始 PatchTST 的 RevIN / patching / TSTiEncoder 等处理流程
      - 在 patch/token 级别做 pooling（mean / max / attn / cls），把 (B, n_tokens, d_model) -> (B, d_model)
      - 通过 projection head 把 d_model 映射到 emb_dim（对比学习中常用）
      - 返回 (B, emb_dim)，可直接用作 chunk-level encoder
    
    输入：
      x: Tensor (batch, sample_nums, seq_len)
    输出：
      emb: Tensor (batch, emb_dim)  （默认 L2 归一化）
    
    设计说明（简洁）：
      - 方便和前面的 ChunkingEncoderV2 配合：Chunking 层会把 (B,S,T) 展平成 (B_flat, S, Tchunk)，
        然后把每个 chunk 送到这里得到固定维度向量 (B_flat, emb_dim)。
      - 保留原始实现的复原（denorm）与 padding 机制，但不输出预测值而是 embedding。
    """
    def __init__(self, cfg):
        super().__init__()
        # 基础参数
        self.device = cfg.device
        cfg_model = cfg.model
        self.c_in = cfg_model.c_in  # 输入通道数（变量数）
        self.context_window = cfg_model.context_window # seq_len

        self.patch_len = cfg_model.patch_len
        self.stride = cfg_model.stride

        self.d_model = cfg_model.d_model
        self.d_ff = cfg_model.d_ff

        self.n_layers = cfg_model.n_layers
        self.n_heads = cfg_model.n_heads

        self.attn_dropout = cfg_model.attn_dropout
        self.dropout = cfg_model.dropout

        self.pe = cfg_model.pe
        self.learn_pe = cfg_model.learn_pe

        self.padding_patch = cfg_model.padding_patch

        self.revin = cfg_model.revin
        self.affine = cfg_model.affine
        self.subtract_last = cfg_model.subtract_last

        self.pool = cfg_model.pool

        self.emb_dim = cfg_model.emb_dim

        self.proj_hidden = cfg_model.proj_hidden
        self.normalize = cfg_model.normalize
        self.use_cls_token = cfg_model.use_cls_token

        self.res_attention = cfg_model.res_attention
        self.pre_norm = cfg_model.pre_norm

        self.out_dim = self.patch_len * ((cfg.light_model.chunking.chunk_len - self.patch_len) // self.stride + 1)  # 计算输出维度

        # === MODIFIED: 保留 RevIN 但用于特征提取路径 ===
        if self.revin:
            # 使用原有 RevIN 接口（行为和原 backbone 保持一致）
            self.revin_layer = RevIN(self.c_in, affine=self.affine, subtract_last=self.subtract_last)

        # === MODIFIED: 保存 patch 参数（与原来的 patch 对齐） ===
        patch_num = int((self.context_window - self.patch_len)/self.stride + 1)
        if self.padding_patch == 'end':
            patch_num += 1
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))

        self.flatten = nn.Flatten(start_dim = -2)

        # === MODIFIED: 使用你原来的 TSTiEncoder（保持实现不变） ===
        # 注意：这里把 max_seq_len 或 q_len 设置为 patch_num，便于 positional encoding 长度对齐
        self.encoder = TSTiEncoder(c_in=self.c_in, patch_num=patch_num, patch_len=self.patch_len,
                                   max_seq_len=patch_num, n_layers=self.n_layers, d_model=self.d_model,
                                   n_heads=self.n_heads, d_k=None, d_v=None, d_ff=self.d_ff, norm='BatchNorm',
                                   attn_dropout=self.attn_dropout, dropout=self.dropout, pe=self.pe, learn_pe=self.learn_pe,
                                   res_attention=self.res_attention, pre_norm=self.pre_norm)

        if self.use_cls_token:
            # cls_token: (1, 1, d_model) -> 在 forward 时 expand 到 (batch, 1, d_model)
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        
        # === MODIFIED: 在 patch_num 维上做 pooling，得到每个变量的表示 (batch, nvars, d_model) ===
        if self.pool == 'attn':
            hidden = max(8, self.d_model // 4)
            self.attn_score = nn.Sequential(
                nn.Linear(self.d_model, max(8, hidden)),
                nn.ReLU(),
                nn.Linear(max(8, hidden), 1)
            )
        elif self.pool in ['mean','max','cls']:
            pass
        else:
            raise ValueError(f"Unsupported pool type: {self.pool}")

        # ========== MODIFIED: projection head（d_model -> emb_dim） ==========
        proj_hidden = self.proj_hidden if self.proj_hidden is not None else max(self.d_model, self.emb_dim)
        self.proj = nn.Sequential(
            nn.Linear(self.d_model, proj_hidden),
            nn.ReLU(),
            nn.Linear(proj_hidden, self.emb_dim)
        )

        self.to(self.device)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError('输入形状应为 (chunk_batch, n_chunks, chunk_len)')
        x = x.to(self.device)
        z = x
        if self.revin:
            z = x.permute(0,2,1)          # (batch, seq_len, c_in)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)          # (batch, c_in, seq_len)
        else:
            z = x

        # === MODIFIED: padding_patch 行为复用 ===
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)

        # === MODIFIED: 执行 unfold -> 得到 patch 矩阵（与原 forward 保持一致） ===
        z_unfold = z.unfold(dimension=-1, size=self.patch_len, step=self.stride).contiguous()
        z_perm = z_unfold.permute(0, 1, 3, 2)  # (batch, c_in, patch_len, patch_num)

        # === MODIFIED: 通过 TSTiEncoder 编码，得到 token-level 输出 ===
        # encoder 返回 z_enc: (batch, nvars, d_model, patch_num)
        z_enc = self.encoder(z_perm)

        # 将 token 维和 d_model 排列为 (batch, nvars, patch_num, d_model)
        z_tokens = z_enc.permute(0,1,3,2)  # (batch, nvars, patch_num, d_model)
        
        # Flatten操作
        z_flatten = self.flatten(z_tokens)  # (batch, nvars, patch_num * d_model)
        return z_flatten
        #  TODO:从这里拆分，下面的合并操作移到上层代码执行，保证backbone的可复用性
        

        # === MODIFIED: 在 patch_num 维上做 pooling，得到每个变量的表示 (batch, nvars, d_model) ===
        if self.pool == 'mean':
            var_repr = z_tokens.mean(dim=2)
        elif self.pool == 'max':
            var_repr = z_tokens.max(dim=2).values
        elif self.pool == 'attn':
            # TODO:
            scores = self.attn_score(z_tokens)  # (batch, nvars, patch_num, 1)
            weights = torch.softmax(scores, dim=2)
            var_repr = (weights * z_tokens).sum(dim=2)
        elif self.pool == 'cls':
            # 在这里简单取第一个 token，若要严格支持 CLS 需在 encoder 输入阶段插入
            var_repr = z_tokens[:, :, 0, :]
        else:
            raise ValueError("Unsupported pool: " + str(self.pool))

        # === MODIFIED: 跨变量聚合（默认取 mean）-> 得到序列级 d_model 表示 (batch, d_model) ===
        # 说明：若你希望对每个变量单独投影/concat，可修改此处逻辑
        seq_repr = var_repr.mean(dim=1)  # (batch, d_model)

        # === MODIFIED: 投影到 emb_dim 并归一化（若需要） ===
        emb = self.proj(seq_repr)  # (batch, emb_dim)
        if self.normalize:
            emb = F.normalize(emb, dim=1)

        return emb


# ============================
# === 使用示例（如何与 ChunkingEncoderV2 配合） ===
# 下面为示例代码，仅作演示用途（不在类内部）。
# ============================

if __name__ == "__main__":
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    from utils.Mypydebug import show_shape
    show_shape()

    config_dir = os.path.join(_ROOT_DIR, 'cl_conf')
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name='pretrain_cfg')
    # === 示例：如何把 PatchTSTFeatureBackbone 作为 chunk encoder 使用 ===

    chunk_len = cfg.light_model.chunking.chunk_len

    # 创建特征提取器（新增类）
    feature_backbone = PatchTST_feat_backbone(cfg=cfg).to(cfg.device)

    # 假造一个 (batch, c_in, seq_len) 的 chunk 输入做前向测试
    batch = 2
    x = torch.randn(batch, 1, chunk_len).to(cfg.device)
    emb = feature_backbone(x)  # (batch, emb_dim)
    print("feature emb shape:", emb.shape)  # 期望 (2, emb_dim)

    # 在实际工程中：ChunkingEncoderV2 会把原始 (B, S, T) -> 平展 -> 生成每个 chunk 的 (B_flat, 1, chunk_len)
    # 然后调用 feature_backbone(sub_chunk) -> 返回 (B_flat, emb_dim)，最后再 reshape 回 (B, S, emb_dim)

# 结束