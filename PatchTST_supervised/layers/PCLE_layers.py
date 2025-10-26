import torch
from torch import nn
from .dilated_conv import DilatedConvEncoder
import numpy as np
from loss.soft_losses import hier_CL_soft

class Patch_soft_CL(nn.Module):
    def __init__(
        self,
        input_dims,                    # 输入特征维度
        output_dims=320,               # 输出表示向量维度
        hidden_dims=128,                # 隐藏层维度
        depth=10,                      # 扩张卷积网络深度
        lambda_=0.5,                   # 实例级与时间级对比学习的权重平衡参数
        tau_temp=0.5,                    # 时间级软对比学习的温度参数
        temporal_unit=0,               # 时间对比学习的最小单元
        soft_instance=False,           # 是否启用实例级软对比学习
        soft_temporal=True,           # 是否启用时间级软对比学习
        feature_extract_net='dilated_conv'    # 特征提取网络类型，默认为TCN
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.depth = depth
        self.lambda_ = lambda_
        self.tau_temp = tau_temp
        self.temporal_unit = temporal_unit
        self.soft_instance = soft_instance
        self.soft_temporal = soft_temporal
        self.feature_extract_net = feature_extract_net

        if self.feature_extract_net == 'dilated_conv':
            self.feature_extract_net = TSEncoder(
                input_dims=input_dims,
                output_dims=output_dims,
                hidden_dims=hidden_dims,
                depth=depth,
                mask_mode='all_true'  # 全部时间步均参与编码
            )
        else:
            pass

    def forward(self, x):
        # input: (B, C, P, N)  B: batch size, C: n_vars, P: patch_len, N: patch_num
        B, C, P, N = x.size()
        x = x.permute(0,1,3,2)  # B x C x N x P
        x = x.reshape(B * C, N, P)  # B*C x N x P

        # ================ 关键：随机裁剪策略 ================
        # 这是TS2Vec的核心思想：从时间序列中裁剪两个重叠的子序列进行对比学习
        ts_l = x.size(1)  # 时间序列长度
        
        # 1. 随机确定重叠区域的长度（crop_l）
        crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
        
        # 2. 随机确定重叠区域在原序列中的位置
        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        
        # 3. 为两个子序列随机扩展边界
        # 第一个子序列：从crop_eleft开始，到crop_right结束
        crop_eleft = np.random.randint(crop_left + 1)
        # 第二个子序列：从crop_left开始，到crop_eright结束  
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        
        # 4. 为每个样本生成随机偏移（支持并行处理）
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
        seq_1 = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft).contiguous()
        seq_2 = take_per_row(x, crop_offset + crop_left,  crop_eright - crop_left).contiguous()

        out1_all = self.feature_extract_net(seq_1)
        # 第二个子序列：较长，包含重叠区域的前半部分
        out2_all = self.feature_extract_net(seq_2)

        # ================ 提取重叠部分进行对比 ================
        # 只对重叠的crop_l长度部分进行对比学习
        out1 = out1_all[:, -crop_l:]  # 第一个子序列的后crop_l个时间步
        out2 = out2_all[:, :crop_l]   # 第二个子序列的前crop_l个时间步

        # 暂时不加入软标签损失,考虑到时间级和实例级的结合
        loss = hier_CL_soft(
                    out1,                        # 第一个子序列的编码
                    out2,                        # 第二个子序列的编码
                    None,           # 软标签矩阵（用于实例级软对比）
                    lambda_=self.lambda_,        # 实例级与时间级损失的权重平衡
                    tau_temp=self.tau_temp,      # 时间级软对比的温度参数
                    temporal_unit=self.temporal_unit,  # 时间对比的最小单元
                    soft_temporal=self.soft_temporal,  # 是否使用时间级软对比
                    soft_instance=self.soft_instance   # 是否使用实例级软对比
                )
        
        x = self.feature_extract_net(x)  # B*C x N x output_dims
        x = x.reshape(B, C, N, -1)  # B x C x N x output_dims

        return x, loss



class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, mask=None):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch
        
        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        
        mask &= nan_mask
        x[~mask] = 0
        
        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co
        
        return x

def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res

def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]