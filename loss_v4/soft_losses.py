import torch
import torch.nn.functional as F
from loss_v4.timelags import *
from loss_v4.hard_losses import *

########################################################################################################
## 软对比学习损失函数
## 核心思想：使用软权重而非硬二分类来计算对比损失，更好地利用样本间的连续相似性关系
########################################################################################################

#------------------------------------------------------------------------------------------#
# (1) 实例级软对比学习 (Instance-wise Soft Contrastive Learning)
# 功能：基于样本间的相似性进行软对比学习，而不是简单的正负样本二分类
#------------------------------------------------------------------------------------------#
def inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R):
    """
    实例级软对比学习损失函数
    
    Args:
        z1, z2: 两个增强样本的特征表示，形状为 [B, T, C]
                B=批次大小, T=时间步数, C=特征维度
        soft_labels_L, soft_labels_R: 软标签矩阵，表示样本间的相似性权重
        
    Returns:
        loss: 实例级软对比损失
        
    原理：
        传统对比学习使用硬标签（正样本权重=1，负样本权重=0）
        软对比学习使用连续的相似性权重，更精细地建模样本关系
    """
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)  # 单样本无法进行对比学习
    
    # 将两个增强样本拼接：[z1; z2] -> [2B, T, C]
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C，转置以便在时间维度上计算
    
    # 计算所有样本对之间的余弦相似度矩阵
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    
    # 构建对比学习的logits：移除对角线元素（自身与自身的相似度）
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # 下三角矩阵（去掉对角线）
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]     # 上三角矩阵（去掉对角线）
    
    # 计算负对数softmax，用于对比学习
    logits = -F.log_softmax(logits, dim=-1)
    
    # 使用软标签计算加权损失
    i = torch.arange(B, device=z1.device)
    loss = torch.sum(logits[:,i]*soft_labels_L)        # z1的损失
    loss += torch.sum(logits[:,B + i]*soft_labels_R)   # z2的损失
    loss /= (2*B*T)  # 归一化
    return loss

#------------------------------------------------------------------------------------------#
# (2) 时间级软对比学习 (Temporal Soft Contrastive Learning)  
# 功能：基于时间邻近性进行软对比学习，捕获时间序列的时间依赖关系
#------------------------------------------------------------------------------------------#
def temp_CL_soft(z1, z2, timelag_L, timelag_R):
    """
    时间级软对比学习损失函数
    
    Args:
        z1, z2: 两个时间子序列的特征表示，形状为 [B, T, C] batch_size, time_steps, feature_dim
        timelag_L, timelag_R: 时间滞后软权重矩阵，表示时间步间的相似性
        
    Returns:
        loss: 时间级软对比损失
        
    原理：
        在时间维度上进行对比学习，时间上相近的位置具有更高的相似性权重
        使用软权重建模时间的连续性，而非硬性的时间对齐
    """
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)  # 单时间步无法进行时间对比
    test_mat1 = z1[1,:,:].squeeze(0)
    logits_test = torch.matmul(test_mat1, test_mat1.transpose(0,1))
    logits_test = -F.log_softmax(logits_test, dim=-1)
    loss_test = torch.sum(logits_test[1,:])
    
    # 在时间维度上拼接两个子序列：[z1, z2] -> [B, 2T, C]
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    
    # 计算时间步之间的相似度矩阵
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    
    # 构建时间对比的logits：移除对角线元素, 构造类似lag的结构, 不含对角线 
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # 下三角矩阵
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]     # 上三角矩阵
    
    # 计算负对数softmax
    logits = -F.log_softmax(logits, dim=-1)
    
    # 使用时间滞后软权重计算加权损失
    t = torch.arange(T, device=z1.device)
    loss = torch.sum(logits[:,t]*timelag_L)        # z1的时间损失
    loss += torch.sum(logits[:,T + t]*timelag_R)   # z2的时间损失
    loss /= (2*B*T)  # 归一化
    return loss

#------------------------------------------------------------------------------------------#
# (3) 层次化软对比学习 (Hierarchical Soft Contrastive Learning)
# 功能：结合实例级和时间级的软对比学习，在多个抽象层次上进行表示学习
# 
# 提供多种时间权重生成策略：
## 3-1) hier_CL_soft : sigmoid衰减
## 3-2) hier_CL_soft_window : 窗口式权重
## 3-3) hier_CL_soft_thres : 阈值式权重
## 3-4) hier_CL_soft_gaussian : 高斯分布权重  
## 3-5) hier_CL_soft_interval : 等间隔权重
## 3-6) hier_CL_soft_wo_inst : 仅时间对比学习（无实例对比）
#------------------------------------------------------------------------------------------#

def hier_CL_soft(z1, z2, soft_labels, tau_temp=2, lambda_=0.5, temporal_unit=0, 
                 soft_temporal=False, soft_instance=False, temporal_hierarchy=True):
    """
    层次化软对比学习：基于sigmoid衰减的多尺度对比学习
    
    Args:
        z1, z2: 两个增强样本的特征表示
        soft_labels: 样本间的软相似性标签矩阵
        tau_temp: 时间级软对比的温度参数，控制时间权重的衰减速度
        lambda_: 实例级损失与时间级损失的平衡参数 (0-1)
        temporal_unit: 开始进行时间对比学习的层级
        soft_temporal: 是否使用时间级软对比学习
        soft_instance: 是否使用实例级软对比学习  
        temporal_hierarchy: 是否使用层次化的时间权重（随深度变化）
        
    Returns:
        loss: 平均的层次化软对比损失
        
    原理：
        1. 在多个时间分辨率层级上进行对比学习
        2. 每个层级都结合实例级和时间级的软对比学习
        3. 使用最大池化逐步降低时间分辨率
        4. 深层使用更宽松的时间权重（temporal_hierarchy=True时）
    """
    
    # 准备软标签矩阵
    if soft_labels is not None:
        soft_labels = torch.tensor(soft_labels, device=z1.device)
        soft_labels_L, soft_labels_R = dup_matrix(soft_labels)
    
    loss = torch.tensor(0., device=z1.device)
    d = 0  # 当前深度层级
    
    # 多尺度对比学习：逐步降低时间分辨率
    while z1.size(1) > 1:  # 当时间序列长度 > 1时继续
        
        # ============ 实例级对比学习 ============
        if lambda_ != 0:  # 如果实例权重非零
            if soft_instance:
                # 使用软实例对比学习
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                # 使用硬实例对比学习
                loss += lambda_ * inst_CL_hard(z1, z2)
                
        # ============ 时间级对比学习 ============        
        if d >= temporal_unit:  # 达到指定层级才开始时间对比
            if 1 - lambda_ != 0:  # 如果时间权重非零
                if soft_temporal:
                    # 生成时间滞后的软权重矩阵
                    if temporal_hierarchy:
                        # 层次化：深层使用更宽松的时间权重
                        timelag = timelag_sigmoid(z1.shape[1], tau_temp*(2**d))
                    else:
                        # 固定：所有层级使用相同的时间权重
                        timelag = timelag_sigmoid(z1.shape[1], tau_temp)
                    timelag = torch.tensor(timelag, device=z1.device)
                    timelag_L, timelag_R = dup_matrix(timelag)
                    loss += (1 - lambda_) * temp_CL_soft(z1, z2, timelag_L, timelag_R)
                else:
                    # 使用硬时间对比学习
                    loss += (1 - lambda_) * temp_CL_hard(z1, z2)
        
        # ============ 降采样到下一个层级 ============
        d += 1
        # 使用最大池化将时间分辨率减半
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)


    # ============ 处理最终层级（时间长度=1） ============
    if z1.size(1) == 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        d += 1

    return loss / d  # 返回平均损失


def hier_CL_soft_window(z1, z2, soft_labels, window_ratio, tau_temp=2, lambda_=0.5,
                        temporal_unit=0, soft_temporal=False, soft_instance=False):
    """
    层次化软对比学习：基于窗口的时间权重生成策略
    
    Args:
        window_ratio: 时间窗口比例，控制时间权重的有效范围
        其他参数同hier_CL_soft
        
    原理：
        使用窗口函数限制时间权重的作用范围，超出窗口的时间步权重为0
        适用于需要局部时间对比的场景
    """
    soft_labels = torch.tensor(soft_labels, device=z1.device)
    soft_labels_L, soft_labels_R = dup_matrix(soft_labels)
    loss = torch.tensor(0., device=z1.device)
    d = 0
    
    while z1.size(1) > 1:
        # 实例级对比学习
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        # 时间级对比学习
        if d >= temporal_unit:
            if 1 - lambda_ != 0:
                if soft_temporal:
                    # 使用窗口式sigmoid权重：在指定窗口内应用sigmoid衰减
                    timelag = timelag_sigmoid_window(z1.shape[1], tau_temp*(2**d), window_ratio)
                    timelag = torch.tensor(timelag, device=z1.device)
                    timelag_L, timelag_R = dup_matrix(timelag)
                    loss += (1 - lambda_) * temp_CL_soft(z1, z2, timelag_L, timelag_R)
                else:
                    loss += (1 - lambda_) * temp_CL_hard(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        d += 1

    return loss / d

def hier_CL_soft_thres(z1, z2, soft_labels, threshold, lambda_=0.5, temporal_unit=0, soft_temporal=False, soft_instance=False):
    """
    层次化软对比学习：基于阈值的时间权重生成策略
    
    Args:
        threshold: 时间距离阈值，超过此阈值的时间步权重为0
        其他参数同hier_CL_soft
        
    原理：
        使用硬阈值截断时间权重，距离小于阈值的时间步权重为1，否则为0
        适用于需要明确时间边界的场景
    """
    soft_labels = torch.tensor(soft_labels, device=z1.device)
    soft_labels_L, soft_labels_R = dup_matrix(soft_labels)
    loss = torch.tensor(0., device=z1.device)
    d = 0
    
    while z1.size(1) > 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        # 时间级对比学习
        if d >= temporal_unit:
            if 1 - lambda_ != 0:
                if soft_temporal:
                    # 使用阈值式权重：超过阈值的时间距离权重为0
                    timelag = timelag_sigmoid_threshold(z1.shape[1], threshold)
                    timelag = torch.tensor(timelag, device=z1.device)
                    timelag_L, timelag_R = dup_matrix(timelag)
                    loss += (1 - lambda_) * temp_CL_soft(z1, z2, timelag_L, timelag_R)
                else:
                    loss += (1 - lambda_) * temp_CL_hard(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        d += 1

    return loss / d


def hier_CL_soft_gaussian(z1, z2, soft_labels, tau_temp=2, lambda_=0.5, temporal_unit=0, soft_temporal=False, soft_instance=False, temporal_hierarchy=True):
    """
    层次化软对比学习：基于高斯分布的时间权重生成策略
    
    Args:
        tau_temp: 高斯分布的标准差参数，控制权重衰减的平滑程度
        temporal_hierarchy: 是否使用层次化的标准差（随深度变化）
        其他参数同hier_CL_soft
        
    原理：
        使用高斯分布生成时间权重，提供比sigmoid更平滑的衰减
        适用于需要平滑时间权重的场景
    """
    soft_labels = torch.tensor(soft_labels, device=z1.device)
    soft_labels_L, soft_labels_R = dup_matrix(soft_labels)
    loss = torch.tensor(0., device=z1.device)
    d = 0
    
    while z1.size(1) > 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        # 时间级对比学习
        if d >= temporal_unit:
            if 1 - lambda_ != 0:
                if soft_temporal:
                    if temporal_hierarchy:
                        # 层次化：深层使用更大的标准差（更平滑的权重）
                        timelag = timelag_gaussian(z1.shape[1], tau_temp/(2**d))
                    else:
                        # 固定：所有层级使用相同的标准差
                        timelag = timelag_gaussian(z1.shape[1], tau_temp)
                    timelag = torch.tensor(timelag, device=z1.device)
                    timelag_L, timelag_R = dup_matrix(timelag)
                    loss += (1 - lambda_) * temp_CL_soft(z1, z2, timelag_L, timelag_R)
                else:
                    loss += (1 - lambda_) * temp_CL_hard(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        d += 1

    return loss / d


def hier_CL_soft_interval(z1, z2, soft_labels, tau_temp=2, lambda_=0.5, temporal_unit=0, soft_temporal=False, soft_instance=False):
    """
    层次化软对比学习：基于等间隔的时间权重生成策略
    
    Args:
        tau_temp: 控制权重衰减的参数（此处用于层次化调整）
        其他参数同hier_CL_soft
        
    原理：
        使用线性衰减生成时间权重，提供均匀的时间相似性分布
        权重 = 1 - |时间距离| / 序列长度
        适用于需要线性时间衰减的场景
    """
    soft_labels = torch.tensor(soft_labels, device=z1.device)
    soft_labels_L, soft_labels_R = dup_matrix(soft_labels)
    loss = torch.tensor(0., device=z1.device)
    d = 0
    
    while z1.size(1) > 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        # 时间级对比学习
        if d >= temporal_unit:
            if 1 - lambda_ != 0:
                if soft_temporal:
                    # 使用等间隔线性衰减权重
                    timelag = timelag_same_interval(z1.shape[1], tau_temp/(2**d))
                    timelag = torch.tensor(timelag, device=z1.device)
                    timelag_L, timelag_R = dup_matrix(timelag)
                    loss += (1 - lambda_) * temp_CL_soft(z1, z2, timelag_L, timelag_R)
                else:
                    loss += (1 - lambda_) * temp_CL_hard(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        d += 1

    return loss / d

def hier_CL_soft_wo_inst(z1, z2, soft_labels, tau_temp=2, lambda_=0.5, temporal_unit=0, soft_temporal=False, soft_instance=False):
    """
    层次化软对比学习：仅时间级对比学习（无实例级对比）
    
    Args:
        参数同hier_CL_soft，但忽略实例级对比学习相关参数
        
    原理：
        专注于时间维度的对比学习，不考虑样本间的相似性
        适用于强调时间依赖性而忽略样本关系的场景
        常用于异常检测等任务
    """
    soft_labels = torch.tensor(soft_labels, device=z1.device)
    soft_labels_L, soft_labels_R = dup_matrix(soft_labels)
    loss = torch.tensor(0., device=z1.device)
    d = 0
    
    while z1.size(1) > 1:
        # 跳过实例级对比学习，仅进行时间级对比学习
        if d >= temporal_unit:
            if 1 - lambda_ != 0:
                if soft_temporal:
                    # 使用sigmoid衰减的时间权重
                    timelag = timelag_sigmoid(z1.shape[1], tau_temp*(2**d))
                    timelag = torch.tensor(timelag, device=z1.device)
                    timelag_L, timelag_R = dup_matrix(timelag)
                    loss += (1 - lambda_) * temp_CL_soft(z1, z2, timelag_L, timelag_R)
                else:
                    loss += (1 - lambda_) * temp_CL_hard(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        # 最终层级也跳过实例级对比学习
        d += 1

    return loss / d


########################################################################################################
## 总结：软对比学习损失函数的优势
##
## 1. 实例级软对比学习：
##    - 传统方法：正样本权重=1，负样本权重=0（硬二分类）
##    - 软对比学习：使用连续的相似性权重，更精细地建模样本关系
##
## 2. 时间级软对比学习：
##    - 传统方法：时间对齐是硬性的（exact matching）
##    - 软对比学习：使用时间邻近性的软权重，建模时间的连续性
##
## 3. 层次化软对比学习：
##    - 在多个时间分辨率上进行对比学习
##    - 浅层捕获细粒度时间模式，深层捕获粗粒度全局模式
##    - 提供多种时间权重生成策略，适应不同的应用场景
##
## 4. 灵活的组合策略：
##    - 可以自由组合实例级和时间级的软对比学习
##    - 通过lambda_参数平衡两种损失的权重
##    - 支持软硬对比学习的混合使用
########################################################################################################

