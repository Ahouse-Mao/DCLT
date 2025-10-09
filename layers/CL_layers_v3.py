import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableDecompose(nn.Module):
    def __init__(self, channels, kernel_size=9, groups=None, init_avg=True):
        """
        channels: 输入通道数(变量数)
        kernel_size: 低通核大小(奇数最好),这样输入与输出长度相同
        groups: groups 参数,默认 groups=channels (每通道独立滤波)
        init_avg: 是否用平均滤波初始化
        """
        super().__init__()
        if groups is None:
            groups = channels
        pad = kernel_size // 2
        self.lowpass = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=pad,
            groups=groups,
            bias=False,
        )
        if init_avg:
            with torch.no_grad():
                self.lowpass.weight.fill_(1.0 / kernel_size)
        self.alpha = nn.Parameter(torch.tensor(1.0))  # 可学习缩放

    def forward(self, x: torch.Tensor):
        # x: (B*N, C, P) 或 (B, C, L) 视使用场景而定
        trend = self.lowpass(x) * self.alpha
        season = x - trend
        return trend, season

class Encoder(nn.Module):
    """
    通道独立的 Transformer Encoder。
    输入: (B*C, d_model, P) — 已经过 conv1d embedding 得到 d_model 通道。
    流程: 先对 Q/K 应用 RoPE,再进行多头自注意力,输出形状保持不变。
    """

    def __init__(self, d_model, n_heads, out_dim, dropout=0.0, rope_base=10000.0, d_ff=None, activation="gelu"):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        head_dim = d_model // n_heads
        assert head_dim % 2 == 0, f"head_dim ({head_dim}) must be even for RoPE"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope_base = rope_base
        self.d_ff = 4 * d_model if d_ff is None else d_ff
        
        # LayerNorm 1 (Pre-Norm)
        self.ln1 = nn.LayerNorm(d_model)

        # 多头注意力的投影(不使用 nn.MultiheadAttention,便于在 Q/K 上插入 RoPE)
        self.W_q = nn.Linear(d_model, d_model, bias=True)
        self.W_k = nn.Linear(d_model, d_model, bias=True)
        self.W_v = nn.Linear(d_model, d_model, bias=True)
        
        # proj
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        # dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

        # LayerNorm 2
        self.ln2 = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward Network
        if activation.lower() == "relu":
            act = nn.ReLU()
        elif activation.lower() == "gelu":
            act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # ffn
        self.ffn = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            act,
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, d_model),
        )

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B', L, E) -> (B', H, L, D)
        Bp, L, E = t.shape
        H, D = self.n_heads, self.head_dim
        return t.view(Bp, L, H, D).transpose(1, 2).contiguous()

    def _merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B', H, L, D) -> (B', L, E)
        Bp, H, L, D = t.shape
        return t.transpose(1, 2).contiguous().view(Bp, L, H * D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B*C, P, d_model) -> 返回: (B*C, P, out_dim)
        """
        BC, P, D = x.shape

        h = x

        # Pre-Norm before attention
        h_norm = self.ln1(h)

        # 线性生成 Q/K/V 并拆头
        q = self._split_heads(self.W_q(h_norm))  # (B', H, P, D)
        k = self._split_heads(self.W_k(h_norm))  # (B', H, P, D)
        v = self._split_heads(self.W_v(h_norm))  # (B', H, P, D)

        # 构建并应用 RoPE 到 Q/K(在点积前)
        cos, sin = build_rope_sin_cos(P, self.head_dim, device=x.device, dtype=x.dtype, base=self.rope_base)
        q, k = apply_rope_to_qk(q, k, cos, sin)  # (B', H, P, D)

        # 多头缩放点积注意力
        attn = F.scaled_dot_product_attention(q, k, v)  # (B', H, P, D)
        attn = self.attn_dropout(attn)

        # 合并头并输出投影
        out = self._merge_heads(attn)  # (B', P, d_model)
        out = self.out_proj(out)  # (B', P, d_model)
        out = self.out_dropout(out)

        # Residual Add
        h = h + out  # (B', P, d_model)

        # Pre-Norm before FFN
        h2 = self.ln2(h)
        h2 = self.ffn(h2)  # (B', P, d_model)
        h2 = self.ffn_dropout(h2)

        # Residual Add
        h = h + h2  # (B', P, d_model)

        return h

class FusionModule(nn.Module):
    def __init__(self, d_model, fusion_type='concat', cross_attn_heads=4, dropout=0.0):
        """
        接受 z_tr, z_se 的形状都是 (Bprime, L, D)
        返回 fused 形状 (Bprime, L, D)
        fusion_type: 'concat' | 'cross_attn_bi' | 'cross_attn_one'
        """
        super().__init__()
        self.fusion_type = fusion_type
        self.d_model = d_model
        if fusion_type == 'concat':
            # concat 沿特征维度 -> 2D -> proj 回 D
            self.proj = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model)
            )
        elif fusion_type in ('cross_attn_bi', 'cross_attn_one'):
            # 使用 batch_first=True 以支持 (Bprime, L, D) 直接输入
            self.mha_tr_on_se = nn.MultiheadAttention(embed_dim=d_model, num_heads=cross_attn_heads, batch_first=True, dropout=dropout)
            if fusion_type == 'cross_attn_bi':
                self.mha_se_on_tr = nn.MultiheadAttention(embed_dim=d_model, num_heads=cross_attn_heads, batch_first=True, dropout=dropout)
            # 投影回 D
            self.proj = nn.Linear((2 if fusion_type=='cross_attn_bi' else 2) * d_model, d_model)
            self.ln = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"Unknown fusion_type {fusion_type}")

    def forward(self, z_tr, z_se):
        """
        z_tr, z_se: (Bprime, N, D)
        returns z_fused: (Bprime, N, D)
        """
        # basic checks can be uncommented in debug
        # assert z_tr.shape == z_se.shape
        if self.fusion_type == 'concat':
            # concat per token
            z_cat = torch.cat([z_tr, z_se], dim=-1)   # (Bprime, N, 2D)
            z = self.proj(z_cat)                      # (Bprime, N, D)
            return z

        elif self.fusion_type == 'cross_attn_one':
            # trend attends to season (single direction)
            # Q=z_tr, K/V=z_se
            tr_att, _ = self.mha_tr_on_se(z_tr, z_se, z_se)  # (Bprime, N, D)
            # residual & fuse with season (or with z_tr)
            z = torch.cat([tr_att + z_tr, z_se], dim=-1)    # (Bprime, N, 2D)
            z = self.ln(self.proj(z))                       # (Bprime, N, D)
            return z

        else:  # cross_attn_bi
            # bi-directional: both attend to each other
            tr_att, _ = self.mha_tr_on_se(z_tr, z_se, z_se)  # trend reads season
            se_att, _ = self.mha_se_on_tr(z_se, z_tr, z_tr)  # season reads trend
            # fuse per token by concat then proj
            z_cat = torch.cat([tr_att, se_att], dim=-1)     # (Bprime, N, 2D)
            z = self.ln(self.proj(z_cat))                   # (Bprime, N, D)
            return z

class TokenProjectionHead(nn.Module):
    """
    把每个 token (每个 patch 的 fused 表示) 投影到对比空间
    输入: z_fused (Bprime, N, 1 or 2, D)
    输出: h (Bprime, N, proj_dim) 且 L2 归一化
    """
    def __init__(self, in_dim, proj_dim=256, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, z):  # z: (Bprime, N, 1 or 2, D)
        Bp, N, K, D = z.shape
        flat = z.view(Bp * N * K, D)
        h = self.net(flat)               # (Bp*N*K, proj_dim)
        h = F.normalize(h, dim=-1)
        h = h.view(Bp, N, K, -1)            # (Bprime, N, K, proj_dim)
        return h

class TokenDecoder(nn.Module):
    """
    Reconstruction Decoder
    用于把token重构回patch,用于计算重构损失
    对比学习的目标是“拉近正样本、推远负样本”,但是这个目标本身并不要求 embedding 保留多少原始序列的细节。
    引入重构模块能够保留足够的原始信息,判别强但信息弱。
    本身整个预训练模型也是作为embdding,后续下游任务的预测也需要保留语义信息

    输入形状: (B*C, N, proj_dim)
    输出形状: (B*C, L)
    """
    def __init__(self, proj_dim, num_patches, seq_len, out_channels, weight_mse, weight_cos):
        super().__init__()
        
        self.proj_dim = proj_dim
        self.num_patches = num_patches
        self.seq_len = seq_len
        self.out_channels = out_channels  # 变量数 C
        self.weight_mse = weight_mse
        self.weight_cos = weight_cos

        self.proj_decrease = nn.Linear(proj_dim, proj_dim // 8)
        self.patch_decrease = nn.Linear(num_patches, num_patches // 2)

        self.decoder = nn.Sequential(
            nn.LayerNorm((proj_dim // 8) * (num_patches // 2)),
            nn.Linear((proj_dim // 8) * (num_patches // 2), seq_len),
            nn.GELU(),
            nn.Linear(seq_len, seq_len)
        )

    def forward(self, z_fused):
        z_fused = z_fused.squeeze(2) if z_fused.dim() == 4 else z_fused  # (Bprime, N, D)
        z = self.proj_decrease(z_fused)  # (Bprime, N, proj_dim/8)
        z = self.patch_decrease(z.transpose(1,2)).transpose(1,2)  # (Bprime, N/2, proj_dim/8)
        z = z.reshape(z.shape[0], -1)  # (Bprime, N/2 * proj_dim/8)
        rec = self.decoder(z)  # (Bprime, N/2 * proj_dim/8) -> (Bprime, seq_len)
        rec = rec.reshape(-1, self.out_channels, self.seq_len)  # (B, C, L)
        return rec

    def rec_loss(self, rec, target):
        """
        计算重构损失
        rec: (B, C, L)
        target: (B, C, L)
        """

        mse_loss = F.mse_loss(rec, target, reduction='mean')

        # 展平为 (B, C*L) 计算整体的余弦相似度
        rec_flat = rec.view(rec.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)
        # 余弦相似度
        cos_sim = F.cosine_similarity(rec_flat, target_flat, dim=1)  # (B,)
        cosine_loss = 1 - cos_sim.mean()
        
        rec_loss = self.weight_mse * mse_loss + self.weight_cos * cosine_loss

        return rec_loss
    
class Soft_CL_weight():
    """
    软对比学习权重分配

    功能：
      - 根据 patch 之间的时间/位置距离,生成 (Q, N) 的权重矩阵,
        其中 Q 是 anchor 的数量(B*C*N),N 是每个变量的 patch 数量。
      - 原始函数为： w(d) = 2 / (1 + exp(dist * sigma)) (distance 越小权重越大)
      - 支持参数：
          sigma: 控制衰减陡峭度(sigma 越大,远处更快衰减,近邻更占优)
          invert: 若 True 则使用 1 - sigmoid(把近优转为远优)
          min_weight: 小权重阈值,低于此值的置 0(节省计算并提高稀疏性)
          normalize: 是否对每行(每个 anchor)在同变量段内部做归一化(使该段内部和为1)
          clamp_max: 为数值稳定,把 exp 的参数裁剪到 [0, clamp_max]
    用法：
      t = TimeLagSigmoid(sigma=1.0, invert=False)
      same_weights = t.compute(n_idx, N, device)  # 返回 (Q, N)
    """
    def __init__(self, sigma=1.0, invert=False, min_weight=1e-6, normalize=False, dtype=torch.float32):
        self.sigma = float(sigma)
        self.invert = bool(invert)
        self.min_weight = float(min_weight)
        self.normalize = bool(normalize)
        self.dtype = dtype

    def compute(self, n_idx, N):
        """
        计算同变量段的权重矩阵
        参数:
          n_idx: torch.LongTensor, shape (Q,), 每个扁平 anchor 对应的 patch 索引 n (0..N-1)
          N: int, 每变量 patch 数量
          device: 设备
          dtype: 返回张量的数据类型
        返回:
          weights_same: torch.Tensor, shape (Q, N), dtype 指定的浮点
            已经把 anchor 自身与左右邻居位置置 0(mask)。
        """
        Q = n_idx.shape[0]
        device = n_idx.device  # 保持与输入索引一致的设备

        # m_idx: (1, N) -> [0,1,...,N-1]
        m_idx = torch.arange(N, device=device, dtype=n_idx.dtype).unsqueeze(0)

        # n_idx_repeat: (Q, 1)
        n_idx_repeat = n_idx.unsqueeze(1)  # (Q, 1)

        # dist: (Q, N) 绝对距离矩阵
        dist = torch.abs(n_idx_repeat - m_idx)  # (Q, N)

        # 用稳定形式计算 timelag_sigmoid：2 * sigmoid(-dist * sigma)
        # 这里不用直接 exp(clamp(...))，使用 sigmoid 更稳定
        same_w = 2.0 * torch.sigmoid(-dist * self.sigma)  # (Q, N)

        # 把非常小的权重置 0（稀疏化）
        if self.min_weight is not None and self.min_weight > 0:
            same_w = torch.where(
                same_w < self.min_weight, torch.zeros_like(same_w), same_w
            )

        # 可选：对同变量段按行归一化（每个 anchor 的行和为 1）
        if self.normalize:
            row_sum = same_w.sum(dim=1, keepdim=True)  # (Q, 1)
            same_w = same_w / (row_sum + 1e-12)

        return same_w.to(device=device, dtype=self.dtype)

def spectral_loss_from_raw(trend_raw, season_raw, cutoff_ratio=0.2):
    """
    频谱损失频谱损失减少了两个 encoder 之间的“语义掠夺”(即一个 encoder 把另一流的频谱成分学习掉),
    从而更容易把 trend/season 做成互补而非冗余的表示。

    trend_raw, season_raw: (B*N, C, P)

    先进行快速傅里叶变换,时域信号转换为频域信号,然后计算功率谱,根据cutoff_ratio频率阈值划分高低频段,
    对趋势信号计算高频能量,对季节信号计算低频能量,
    因为趋势信号理想上应该是低频主导的(平滑、无快速波动),所以高频能量越高,损失越大。
    因为季节信号理想上应该是高频主导的(周期性振荡),所以低频能量越高,损失越大。
    最后将两者相加作为损失值返回。
    我们先对 channels 取平均能量,然后 rfft 沿 time dim (P)
    cutoff_ratio: frequency cutoff fraction for low/high band
    """
    # compute rfft 快速傅里叶变换
    Tf = torch.fft.rfft(trend_raw, dim=-1)   # (B*N, C, F)
    Sf = torch.fft.rfft(season_raw, dim=-1)
    # power per patch averaged over channels
    power_T = (Tf.real ** 2 + Tf.imag ** 2).mean(dim=1)  # (B*N, F)
    power_S = (Sf.real ** 2 + Sf.imag ** 2).mean(dim=1)
    Fnum = power_T.shape[-1]
    cutoff = max(1, int(Fnum * cutoff_ratio))
    # trend: penalize high freq energy
    trend_high = power_T[:, cutoff:].mean()
    # season: penalize low freq energy
    season_low = power_S[:, :cutoff].mean()
    
    return trend_high + season_low

def token_ortho_loss(z_tr, z_se):
    """
    正交损失对两个编码子空间施加约束, 目标是让它们再向量空间中不重叠,不冗余,
    (即正交或低余弦相似度),以增强表示的可区分性和可组合性。

    z_tr, z_se: (Bprime, N, D)
    返回标量,鼓励每个 token 的 trend/season 表示在方向上不冗余
    """
    # flatten tokens
    Bp, N, D = z_tr.shape
    tr_flat = z_tr.view(Bp * N, D)
    se_flat = z_se.view(Bp * N, D)
    cos = F.cosine_similarity(tr_flat, se_flat, dim=-1)  # (Bp*N,)
    return (cos ** 2).mean()






# ===== Rotary Positional Embedding (RoPE) utilities =====

def build_rope_sin_cos(
    seq_len: int, head_dim: int, device, dtype, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    构建 RoPE 所需的 cos/sin 缓存。
    返回:
      cos, sin: 形状 (1, 1, seq_len, head_dim)
    """
    assert head_dim % 2 == 0, "head_dim 必须为偶数以便两两配对旋转"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))  # (D/2,)
    t = torch.arange(seq_len, device=device, dtype=dtype)  # (L,)
    freqs = torch.einsum("l,d->ld", t, inv_freq)  # (L, D/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (L, D)
    cos = emb.cos().unsqueeze(0).unsqueeze(0)  # (1,1,L,D)
    sin = emb.sin().unsqueeze(0).unsqueeze(0)  # (1,1,L,D)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    将最后一维偶/奇通道两两配对做 90° 旋转: [x_even, x_odd] -> [-x_odd, x_even]
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    x_rot = torch.stack((-x_odd, x_even), dim=-1)
    return x_rot.flatten(-2)  # 恢复到原 head_dim


def apply_rope_to_qk(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对 q,k 应用 RoPE。
    期望形状:
      q, k : (B, H, L, D)
      cos, sin : (1, 1, L, D)
    返回:
      q_rope, k_rope (同形状)
    """
    q_rope = (q * cos) + (rotate_half(q) * sin)
    k_rope = (k * cos) + (rotate_half(k) * sin)
    return q_rope, k_rope











