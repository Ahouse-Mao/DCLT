import torch
import torch.nn.functional as F

from utils.Mypydebug import show_shape
show_shape()

def multi_pos_contrastive_loss(
    emb: torch.Tensor,
    group: torch.Tensor = None,
    temperature: float = 0.1,
    normalize: bool = True,
    reduction: str = "mean",  # 'mean' 或 'sum'
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    多正样本对比损失（向量化实现）
    论文中的公式：对每个 anchor i，对其所有正样本 n ∈ P(i) 计算
      L_i = (1/|P(i)|) * sum_{n in P(i)} -log( exp(sim(i,n)/tau) / sum_{k not in P(i)} exp(sim(i,k)/tau) )
    然后对所有 anchor 做 reduction（mean 或 sum）。

    参数:
        emb: (B, P, D) 或 (M, D) 的 embedding 张量
             - (B, P, D) 时：B = 原始样本数，P = 每个样本的增强/视图数，D = 嵌入维度
             - (M, D) 时：扁平化的 embeddings，需同时提供 group（长度为 M）
        group: 当 emb 为 (M, D) 时必须提供。长度 M，表示每个 embedding 属于哪个原始样本（0..B-1）
        temperature: 温度系数 τ
        normalize: 是否先做 L2 归一化（以得到余弦相似度）
        reduction: 'mean' 或 'sum'，对所有 anchor 的 loss 做归约
        eps: 防止除零的小常数

    返回:
        标量损失 tensor
    """
    # ===========================
    # 1) 准备扁平化的 embeddings 和分组信息
    # ===========================
    if emb.dim() == 3:
        # 输入为 (B, P, D)
        B, P, D = emb.shape
        M = B * P
        emb_flat = emb.reshape(M, D)  # 扁平化为 (M, D)
        if group is None:
            # group 表示每个视图属于哪一个原始样本，长度 M
            group = torch.arange(B, device=emb.device, dtype=torch.long).repeat_interleave(P)
    elif emb.dim() == 2:
        # 输入为 (M, D)
        emb_flat = emb
        M, D = emb_flat.shape
        if group is None:
            raise ValueError("当 emb 为 (M, D) 时，必须提供 `group` 张量。")
        if group.numel() != M:
            raise ValueError("group 的长度必须等于 embeddings 的数量 M。")
    else:
        raise ValueError("emb 必须是 2D 或 3D 张量。")

    # ===========================
    # 2) 归一化（若需要），然后计算相似度矩阵 S (M x M)
    #    若归一化则点积等价于余弦相似度
    # ===========================
    if normalize:
        emb_flat = F.normalize(emb_flat, p=2, dim=1, eps=eps)

    # 相似度矩阵：每对向量的点积（若归一化即为余弦相似度）
    S = emb_flat @ emb_flat.t()  # (M, M)
    S = S / float(temperature)   # 除以温度 tau（等价于 /tau）

    device = emb_flat.device

    # ===========================
    # 3) 构造正样本掩码 mask_pos，和去除自我对角线后的正样本掩码 mask_pos_no_self
    #    mask_pos[i,j] = True 当且仅当 i 和 j 来自同一原始样本（包括自身）
    # ===========================
    g = group.view(-1, 1)
    mask_pos = (g == g.t())            # 包含对角线（self）
    diag = torch.eye(S.size(0), dtype=torch.bool, device=device)
    mask_pos_no_self = mask_pos & (~diag)  # 正样本位置（排除自身）

    # ===========================
    # 4) 为分母计算 logsumexp，分母只对“负样本”（不同组）求和
    #    所以把正样本（包括 self）对应位置设为 -inf，使其在 logsumexp 中不被计入
    # ===========================
    neg_mask = ~mask_pos  # 不同组的位置为 True，即负样本
    NEG_INF = -1e9
    s_for_denom = S.clone()
    s_for_denom[~neg_mask] = NEG_INF  # 将非负样本（正样本+self）置为 -inf

    # denom_log 对每个 anchor i 计算 log(sum_{k ∈ negatives} exp(s_{i,k}))
    denom_log = torch.logsumexp(s_for_denom, dim=1)  # (M,)

    # ===========================
    # 5) 对于每个正样本对 (i,j)，对应的 pair loss 为: -s_{i,j} + denom_log[i]
    #    我们构造一个完整的 loss_matrix 然后只保留正样本位置
    # ===========================
    loss_matrix = -S + denom_log.view(-1, 1)  # (M, M)，广播 denom_log

    # pos_mask_float 用于求和（只保留正样本位置）
    pos_mask_float = mask_pos_no_self.float()
    pos_counts = pos_mask_float.sum(dim=1)  # 每个 anchor 的正样本数（理论上若均匀为 P-1）
    # 为了避免除零（某些 anchor 可能没有正样本），把 0 替换为 1（但后面会把这些 anchor 的贡献设为 0）
    safe_pos_counts = pos_counts.clone()
    safe_pos_counts[safe_pos_counts == 0] = 1.0

    # 每个 anchor 的正样本对的 loss 之和
    sum_loss_per_anchor = (loss_matrix * pos_mask_float).sum(dim=1)  # (M,)
    # 对该 anchor 上的正样本取均值
    mean_loss_per_anchor = sum_loss_per_anchor / safe_pos_counts  # (M,)

    # 如果某 anchor 没有正样本，则把它的 loss 设为 0（乘以 mask）
    mean_loss_per_anchor = mean_loss_per_anchor * (pos_counts > 0).float()

    # ===========================
    # 6) 最后 reduction
    # ===========================
    if reduction == "mean":
        return mean_loss_per_anchor.mean()
    elif reduction == "sum":
        return mean_loss_per_anchor.sum()
    else:
        raise ValueError("reduction 必须是 'mean' 或 'sum'。")


# ===========================
# 使用示例（含中文说明）
# ===========================
if __name__ == "__main__":
    # 示例1：B=4 个原始样本，每个样本 P=3 个视图，嵌入维度 D=128
    B, P, D = 4, 3, 128
    emb = torch.randn(B, P, D)  # 默认在 CPU 上，训练时可 .cuda()
    loss = multi_pos_contrastive_loss(emb, temperature=0.1, reduction='mean')
    print("loss:", loss.item())

    # 示例2：扁平化输入，需要提供 group
    emb_flat = emb.view(B * P, D)
    group = torch.arange(B).repeat_interleave(P)  # 长度 M=B*P，表示每个视图所属的原始样本 id
    loss2 = multi_pos_contrastive_loss(emb_flat, group=group, temperature=0.1)
    print("loss2:", loss2.item())

    # 验证两种调用方式等价
    assert torch.allclose(loss, loss2, atol=1e-6)
