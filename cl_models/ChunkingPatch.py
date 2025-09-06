import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Dict
from omegaconf import DictConfig

import os, sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from cl_models.PatchTST_feat_backbone import PatchTST_feat_backbone

class ChunkingPatch(nn.Module):
    """
    支持 DataLoader 返回 (B, S, T) 的统一 chunking 编码器。
    - anchors: (B, 1, T)
    - pos:     (B, P, T)
    - neg:     (B, N, T)

    输出:
      anchor -> (B, emb_dim)    (S==1 自动 squeeze)
      pos    -> (B, P, emb_dim)
      neg    -> (B, N, emb_dim)
    """

    def __init__(self, cfg: Optional[DictConfig]):
        super().__init__()
        self.chunk = cfg.light_model.chunking
        self.chunk_len = self.chunk.chunk_len
        self.overlap = self.chunk.overlap
        self.aggregator = self.chunk.aggregator
        self.transformer_num_layers = self.chunk.transformer_num_layers
        self.lstm_hidden = self.chunk.lstm_hidden
        self.device = cfg.device
        self.emb_dim = cfg.model.emb_dim  # 直接从 model 配置里取 emb_dim
        self.max_chunk_batch = self.chunk.max_chunk_batch # 每次送入 backbone 的 chunk 数量，防OOM

        self.backbone = PatchTST_feat_backbone(cfg) # 传入参数给patchtstbackbone

        # assert 0.0 <= overlap < 1.0, "overlap must be in [0,1)"
        # self.overlap = float(overlap)
        # self.max_chunk_batch = int(max_chunk_batch)
        # self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        # if emb_dim is None:
        #     emb_dim = getattr(backbone, 'embedding_dim', None)
        #     if emb_dim is None:
        #         raise ValueError("请提供 emb_dim 或保证 backbone 包含属性 embedding_dim")
        # self.emb_dim = int(emb_dim)

        # self.aggregator_type = aggregator
        if self.aggregator == 'mean':
            self.aggregator = None
        elif self.aggregator == 'attn':
            self.aggregator = nn.Sequential(
                nn.Linear(self.emb_dim, max(8, self.emb_dim // 4)),
                nn.ReLU(),
                nn.Linear(max(8, self.emb_dim // 4), 1)
            )
        elif self.aggregator == 'transformer':
            layer = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=max(1, min(8, self.emb_dim//64)))
            self.aggregator = nn.TransformerEncoder(layer, num_layers=self.transformer_num_layers)
        elif self.aggregator == 'lstm':
            hidden = self.lstm_hidden or max(8, self.emb_dim // 2)
            self.aggregator = nn.LSTM(input_size=self.emb_dim, hidden_size=hidden, num_layers=1, batch_first=True, bidirectional=True)
            self._lstm_proj = nn.Linear(hidden*2, self.emb_dim)
        else:
            raise ValueError("unsupported aggregator: " + str(self.aggregator))

        self.to(self.device)

    # -------------------------
    # 用于生成chunks序列
    # sliding window (保留 channel)：输入 (B_S, T) -> 返回 (B_S, n_chunks, chunk_len)
    # -------------------------
    def _make_chunks_(self, x: torch.Tensor) -> torch.Tensor:
        """把二维序列划分为重叠窗口。
        输入:  x (B_S, T)
        输出:  chunks (B_S, n_chunks, chunk_len)
        说明:  仅处理单通道/已展平后的情况; stride = chunk_len * (1-overlap)
        """
        if x.dim() != 2:
            raise ValueError(f"_make_chunks_ 期望输入 (B_S,T) 2D 张量, 得到 {tuple(x.shape)}")
        B_S, T = x.shape
        stride = int(self.chunk_len * (1 - float(self.overlap)))
        if stride <= 0:
            raise ValueError(f"overlap={self.overlap} 导致 stride<=0; 请减小 overlap 或增大 chunk_len")

        # 不足一个 chunk: 右侧 pad
        if T < self.chunk_len:
            pad_len = self.chunk_len - T
            x = F.pad(x, (0, pad_len), value=0.0)
            T = x.shape[1]
        else:
            # 让尾部对齐: (T - chunk_len) 能被 stride 整除
            remainder = (T - self.chunk_len) % stride
            if remainder != 0:
                pad_len = stride - remainder
                x = F.pad(x, (0, pad_len), value=0.0)
                T = x.shape[1]

        # unfold 沿时间维 dimension=1 -> (B_flat, n_chunks, chunk_len)
        chunks = x.unfold(dimension=1, size=self.chunk_len, step=stride).contiguous()
        return chunks

    # -------------------------
    # chunks进行backbone编码的过程
    # 编码平展序列 (B_flat, C, T) -> (B_flat, n_chunks, chunk_len)
    # -------------------------
    def _chunks_process(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        sequences: (B_S, T)
        returns:   (B_S, n_chunks, chunk_len)
        """
        sequences = sequences.to(self.device)
        B_S, T = sequences.shape
        chunks = self._make_chunks_(sequences)  # (B_S, n_chunks, chunk_len)
        B_S, n_chunks, chunk_len = chunks.shape

        # 调整为 (B_S * n_chunks, chunk_len)
        # currently (B_S n_chunks, chunk_len) -> permute -> (B_S n_chunks, chunk_len) -> view
        # x = chunks.permute(0, 2, 1).contiguous().view(B_S * n_chunks, chunk_len)  # (B_S*n_chunks, chunk_len)

        # 分小批次通过 backbone（防OOM）
        out_parts = []
        total = chunks.shape[0]
        idx = 0
        while idx < total: # 通过max_chunk_batch来控制每次进入的数量
            j = min(idx + self.max_chunk_batch, total)
            sub = chunks[idx:j].to(self.device)  # (chunk_batch, n_chunks, chunk_len)
            emb = self.backbone(sub)  # expect (sub_b, n_chunks, chunk_len)
            # if emb.dim() == 3:
            #     # backbone 返回 token-level, pool到 vector
            #     emb = emb.mean(dim=1)
            out_parts.append(emb)
            idx = j
        emb_all = torch.cat(out_parts, dim=0)  # (B_S * n_chunks, n_chunks, chunk_len)

        return emb_all
        
        # TODO:聚合也不在这一步进行，移动至上层执行
        # 聚合
        emb_seq = self._aggregate_chunk_embeddings(emb_chunks)  # (B_S, emb_dim)
        return emb_seq

    # # -------------------------
    # # 聚合实现(已被废弃)
    # # -------------------------
    # def _aggregate_chunk_embeddings(self, emb_chunks: torch.Tensor) -> torch.Tensor:
    #     # emb_chunks: (B, n_chunks, emb_dim)
    #     B, n_chunks, D = emb_chunks.shape
    #     if self.aggregator_type == 'mean':
    #         return emb_chunks.mean(dim=1)
    #     elif self.aggregator_type == 'attn':
    #         scores = self.aggregator(emb_chunks)  # (B, n_chunks, 1)
    #         weights = torch.softmax(scores, dim=1)
    #         out = (weights * emb_chunks).sum(dim=1)
    #         return out
    #     elif self.aggregator_type == 'transformer':
    #         # transformer expects (seq_len, batch, d_model)
    #         x = emb_chunks.permute(1, 0, 2)  # (n_chunks, B, D)
    #         y = self.aggregator(x)  # (n_chunks, B, D)
    #         out = y.mean(dim=0)     # (B, D)
    #         return out
    #     elif self.aggregator_type == 'lstm':
    #         out, _ = self.aggregator(emb_chunks)  # (B, n_chunks, hidden*2)
    #         pooled = out.mean(dim=1)              # (B, hidden*2)
    #         projected = self._lstm_proj(pooled)  # (B, D)
    #         return projected
    #     else:
    #         raise RuntimeError("unsupported aggregator")

    # -------------------------
    # 外部输入的形状变化
    # 处理外部输入 (B, S, T) -> 返回 (B, S, n_chunks, backbone_outer_dim)
    # -------------------------
    def _emb_process(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        x: None or (B, S, T)  (S can be 1 for anchors)
        returns: None or (B, S, n_chunks, backbone_outer_dim)
        """
        B, S, T = x.shape
        # 把 (B, S, T) 转成 (B*S, T)
        x_flat = x.view(B * S, T)
        # 编码 -> (B*S, n_chunks, backbone_outer_dim)
        emb_chunks = self._chunks_process(x_flat)  # (B*S, n_chunks, backbone_outer_dim)
        # 恢复为 (B, S, n_chunks, backbone_outer_dim)
        emb = emb_chunks.view(B, S, emb_chunks.shape[-2], emb_chunks.shape[-1])

        return emb  # (B, S, n_chunks, backbone_outer_dim)

    # -------------------------
    # 前向
    # -------------------------
    def forward(self, anchors: torch.Tensor, pos: Optional[torch.Tensor] = None, neg: Optional[torch.Tensor] = None) -> Dict[str, Optional[torch.Tensor]]:
        """
        支持仅输入 anchor(推理模式), pos 或 neg 可以为 None。
        约定:
          - anchors: (B, 1, T)
          - pos:     (B, P, T) 或 None
          - neg:     (B, N, T) 或 None
        返回字典的键始终包含 'anchor','pos','neg'，当对应输入为 None 时返回值为 None。
        """

        B, A, T = anchors.shape  # A 应该为 1
        parts = [anchors]
        sizes = [A]
        if pos is not None:
            parts.append(pos)
            sizes.append(pos.shape[1])
        if neg is not None:
            parts.append(neg)
            sizes.append(neg.shape[1])

        combined = torch.cat(parts, dim=1).to(self.device)  # (B, A + (P?) + (N?), T)

        # 统一编码
        emb_combined = self._emb_process(combined)  # 返回(B, S_total, emb_dim)

        # 按 sizes 切分
        offsets = []
        acc = 0
        for s in sizes:
            offsets.append((acc, acc + s))
            acc += s

        # anchor
        a0, a1 = offsets[0]
        anchor_emb = emb_combined[:, a0:a1, :]

        idx = 1
        pos_emb = None
        neg_emb = None
        if pos is not None:
            p0, p1 = offsets[idx]
            pos_emb = emb_combined[:, p0:p1, :]
            idx += 1
        if neg is not None:
            n0, n1 = offsets[idx]
            neg_emb = emb_combined[:, n0:n1, :]

        return {'anchor': anchor_emb, 'pos': pos_emb, 'neg': neg_emb}


if __name__ == "__main__":
    from data_provider.DCLT_data_loader_v2 import GraphContrastDataset
    from torch.utils.data import DataLoader

    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    from utils.Mypydebug import show_shape
    show_shape()

    config_dir = os.path.join(_ROOT_DIR, 'cl_conf')
    with initialize_config_dir(config_dir=config_dir, version_base=None):
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

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=dataset.collate_fn)

    for anchors, pos, neg in dataloader:
        anchors = anchors.to(chunk_model.device)
        pos = pos.to(chunk_model.device) if pos is not None else None
        neg = neg.to(chunk_model.device) if neg is not None else None
        out = chunk_model(anchors, pos, neg)