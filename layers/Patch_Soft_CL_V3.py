import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Patch_Soft_CL(nn.Module):
    def __init__(self, cfg, patch_num):
        super(Patch_Soft_CL, self).__init__()
        # soft_cl_ctrl
        self.use_instance_cl = cfg.model.soft_cl_ctrl.use_instance_cl
        self.use_temporal_cl = cfg.model.soft_cl_ctrl.use_temporal_cl
        self.cl_weight = cfg.model.soft_cl_ctrl.cl_weight
        self.weight_mode = cfg.model.soft_cl_ctrl.weight_mode
        

        # soft_cl_params
        self.patch_sim_weight = cfg.model.soft_cl_params.patch_sim_weight
        self.tau_temporal = cfg.model.soft_cl_params.tau_temporal
        self.patch_len = cfg.model.patch_len
        self.stride = cfg.model.stride
        self.patch_num = patch_num
        self.eps = 1e-8
    
    def forward(self, z1, z2):
        # z1, (B_prime, N, D)
        # z2, (B_prime, N, K, D)
        K = z2.shape[2]
        if K == 1:
            z2 = z2.squeeze(2) # (B_prime, N, D)
            soft_weight = self.generate_weight(z1.shape[1])
            soft_weight = torch.tensor(soft_weight, device=z1.device)
            soft_weight = self.dup_matrix(soft_weight, K)
            temporal_loss = self.temporal_cl(z1, z2, soft_weight, K)
        elif K == 2:
            soft_weight = self.generate_weight(z1.shape[1])
            soft_weight = torch.tensor(soft_weight, device=z1.device)
            soft_weight = self.dup_matrix(soft_weight, K)
            temporal_loss = self.temporal_cl(z1, z2, soft_weight, K)
        return temporal_loss * self.cl_weight
    
    def inst_cl(self, z1, z2):
        pass


    def temporal_cl(self, z1, z2, soft_weight, K):
        # z1 = F.normalize(z1, dim=-1) # L2归一化
        # z2 = F.normalize(z2, dim=-1) # L2归一化
        if K == 1:
            soft_weight_L, soft_weight_R = soft_weight
            B, T, D = z1.shape

            # 时间维度拼接2个矩阵
            z = torch.cat([z1, z2], dim=1) # (B, 2T, D)

            # 计算所有样本间的相似度矩阵
            sim = torch.matmul(z, z.transpose(1, 2)) # (B, 2T, 2T)

            # 构建对比学习logits, 移除对角线元素, 匹配上权重矩阵
            logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # 下三角矩阵（去掉对角线）
            logits += torch.triu(sim, diagonal=1)[:, :, 1:]     # 上三角矩阵（去掉对角线）(B, 2T, 2T-1)

            # 计算负对数softmax，用于对比学习
            logits = -F.log_softmax(logits, dim=-1)
            
            # 使用软标签计算加权损失
            i = torch.arange(B, device=z1.device)
            loss = torch.sum(logits[:,i]*soft_weight_L)        # z1的损失
            loss += torch.sum(logits[:,B + i]*soft_weight_R)   # z2的损失
            loss /= (2*B*T)  # 归一化
            return loss
        if K == 2:
            soft_weight_1, soft_weight_2, soft_weight_3 = soft_weight
            B, N, D = z1.shape
            test_mat1 = z1[1,:,:].squeeze(0)
            logits_test = torch.matmul(test_mat1, test_mat1.transpose(0,1))
            logits_test = -F.log_softmax(logits_test, dim=-1)
            loss_test = torch.sum(logits_test[1,:])
            # 时间维度拼接3个矩阵
            z = torch.cat([z1, z2[:, :, 0, :], z2[:, :, 1, :]], dim=1) # (B, 3T, D)

            # 计算所有样本间的相似度矩阵
            sim = torch.matmul(z, z.transpose(1, 2)) # (B, 3T, 3T)

            # 构建对比学习logits, 移除对角线元素, 匹配上权重矩阵
            logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # 下三角矩阵（去掉对角线）
            logits += torch.triu(sim, diagonal=1)[:, :, 1:]     # 上三角矩阵（去掉对角线）(B, 3T, 3T-1)

            # 计算负对数softmax，用于对比学习
            logits = -F.log_softmax(logits, dim=-1)
            
            # 使用软标签计算加权损失
            i = torch.arange(N, device=z1.device)

            loss = torch.sum(logits[:,i]*soft_weight_1)#/(3*N)        # z1的损失, 这里选择sum是因为每个样本，都要计算自己与同变量其它patch的对比损失
            # logits_1 = logits[:, i]*soft_weight_2
            # 这里可以这样理解, 首先B就当作batch_size, 尽管这里B是由于B,C聚合起来的, 但是通道独立处理，这里直接聚合
            # 然后N相当于原始softCLT中的T(时间步), 然后logits的形状是(B, 3N, 3N-1), 相当于得到了一个变量内所有patch间的对比学习logits
            # 每个logits[1]里的矩阵，每一行都是该patch与其它patch，及其增强样本的对比学习损失
            loss += torch.sum(logits[:,N + i]*soft_weight_2)#/(3*N)   # z2第一个增强样本的损失
            loss += torch.sum(logits[:,2*N + i]*soft_weight_3)#/(3*N) # z2第二个增强样本的损失
            loss /= (3*B*N)  # 归一化
            return loss

    def generate_weight(self, T):
        dist = np.arange(T)
        dist = np.abs(dist - dist[:, np.newaxis])
        dist_sim = 2 / (1 + np.exp(dist * self.tau_temporal))
        if self.weight_mode == "patch_and_dist_sim":
            patch_sim = (self.patch_len - dist * self.stride) / self.patch_len
            patch_sim = np.where(patch_sim < 0, 0, patch_sim)  # 把完全没有重叠度的patch设为0
            patch_sim = np.where(patch_sim == 0, dist_sim, patch_sim)
            matrix = dist_sim
            matrix = patch_sim * self.patch_sim_weight + matrix * (1 - self.patch_sim_weight)
        elif self.weight_mode == "patch_sim":
            patch_sim = (self.patch_len - dist * self.stride) / self.patch_len
            patch_sim = np.where(patch_sim < 0, 0, patch_sim)  # 把完全没有重叠度的patch设为0
            patch_sim = np.where(patch_sim == 0, dist_sim, patch_sim)
            matrix = patch_sim
        elif self.weight_mode == "dist_sim":
            matrix = dist_sim
        matrix = np.where(matrix < 1e-6, 0, matrix)  # set very small values to 0         
        return matrix

    def dup_matrix(self, mat, K):
        if K == 1:
            mat0 = torch.tril(mat, diagonal=-1)[:, :-1] # 取下三角部分，不含对角，去掉最后一列
            mat0 += torch.triu(mat, diagonal=1)[:, 1:] # 取上三角部分，不含对角，去掉第一列，加到下三角部分，合成为仅去掉对角线的矩阵
            mat1 = torch.cat([mat0,mat],dim=1) # mat0在前，mat1在后
            mat2 = torch.cat([mat,mat0],dim=1) # mat在前，mat0在后, 两种拼接方式，分别表示2个子序列的weight
            return [mat1, mat2] # 2个其实可以拼成一个大矩阵, (19+19, 37), 相当于得到了每一个增强样本, 与除自己及自己的增强样本外的所有样本的logits
        elif K == 2:
            mat0 = torch.tril(mat, diagonal=-1)[:, :-1] # 取下三角部分，不含对角，去掉最后一列
            mat0 += torch.triu(mat, diagonal=1)[:, 1:] # 取上三角部分，不含对角，去掉第一列，加到下三角部分，合成为仅去掉对角线的矩阵
            mat1 = torch.cat([mat0,mat,mat],dim=1)
            mat2 = torch.cat([mat,mat0,mat],dim=1)
            mat3 = torch.cat([mat,mat,mat0],dim=1)
            return [mat1, mat2, mat3] # 3个其实可以拼成一个
    


