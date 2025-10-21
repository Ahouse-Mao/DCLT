import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from loss_v4.soft_losses import *
import numpy as np
from models_v4.encoder import TSEncoder
from loss_v4.hard_losses import *
from loss_v4.soft_losses import *
from utils_v4.utils import *

from tasks_v4 import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score

from torch.utils.tensorboard import SummaryWriter
from layers.RevIN import RevIN
'''
ver1:
思路:
把patch视作timestamp, 然后后续的对比学习和ts2vec一样进行处理
操作:
1. 对时间序列进行patch划分, 得到形状为 (B, C, N, P) 的张量, 其中N为patch数量, P为patch长度
2. 将patch维度N视作时间步, P视作特征维度, 送入进行卷积, [这里考虑一下是否需要C和N合并, 一起送入卷积, 一起送入可以混合不同通道的patch信息]
3. 卷积输出后, 得到形状为 (B, C, N, D)
4. 将patch维度N视作时间步, D视作特征维度, 进行后续的对比学习
'''

class TS2Vec:
    '''
    TS2Vec模型: 时间序列的软对比学习表示学习模型
    
    该模型通过以下方式进行自监督学习:
    1. 对时间序列进行随机裁剪，生成两个重叠的子序列
    2. 使用扩张卷积编码器对子序列进行编码
    3. 在重叠部分进行层次化软对比学习
    4. 支持实例级和时间级的软对比学习

    注意:
    1. ver1当前版本没有融合times_features
    '''
    
    def __init__(
        self,
        patch_cl_ver,
        input_dims,                    # 输入特征维度
        n_vars=None,
        output_dims=320,               # 输出表示向量维度
        hidden_dims=64,                # 隐藏层维度
        depth=10,                      # 扩张卷积网络深度
        device='cuda',                 # 计算设备
        lr=0.001,                      # 学习率
        batch_size=16,                 # 批次大小
        lambda_=0.5,                   # 实例级与时间级对比学习的权重平衡参数
        tau_temp=2,                    # 时间级软对比学习的温度参数
        max_train_length=None,         # 最大训练序列长度
        temporal_unit=0,               # 时间对比学习的最小单元
        after_iter_callback=None,      # 每次迭代后的回调函数
        after_epoch_callback=None,     # 每个epoch后的回调函数
        soft_instance=False,           # 是否启用实例级软对比学习
        soft_temporal=False,           # 是否启用时间级软对比学习
        patch_len=None,                # patch长度
        patch_stride=None,             # patch步幅
        padding_patch="end",           # patch填充方式
        revin=True                 # 是否使用RevIN归一化
    ):
        """
        初始化TS2Vec模型
        
        Args:
            input_dims: 输入时间序列的特征维度
            output_dims: 编码后的表示向量维度，默认320
            hidden_dims: 网络隐藏层维度，默认64
            depth: 扩张卷积网络的层数，默认10层
            device: 训练设备，'cuda'或'cpu'
            lr: 学习率
            batch_size: 训练批次大小
            lambda_: 平衡实例级和时间级对比损失的权重参数
            tau_temp: 时间级软对比学习中控制相似性衰减的温度参数
            max_train_length: 如果序列过长，将被裁剪到此长度
            temporal_unit: 时间对比学习的最小时间单元
            soft_instance: 是否使用软实例对比学习（基于样本相似性）
            soft_temporal: 是否使用软时间对比学习（基于时间邻近性）
        """
        
        super().__init__()
        # 基础训练参数
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.tau_temp = tau_temp
        self.lambda_ = lambda_
        
        # 训练配置参数
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        
        # 构建编码器网络：使用扩张卷积进行特征提取
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, 
                             hidden_dims=hidden_dims, depth=depth).to(self.device)
        
        # 指数移动平均网络：用于稳定训练和推理
        # SWA (Stochastic Weight Averaging) 提供更平滑的模型权重
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)
        
        # 回调函数
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        # 软对比学习配置
        self.soft_instance = soft_instance  # 是否使用基于样本相似性的软对比
        self.soft_temporal = soft_temporal  # 是否使用基于时间邻近性的软对比
        
        # 训练状态追踪
        self.n_epochs = 0   # 已训练的epoch数
        self.n_iters = 0    # 已训练的迭代数

        # 额外增加模块
        # RevIn
        self.revin = revin
        if self.revin: 
            if n_vars is None:
                raise ValueError("当启用RevIN时，必须提供 n_vars 参数")
            self.revin_layer = RevIN(n_vars, affine=False, subtract_last=0).to(self.device)

        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.padding_patch = padding_patch
        if self.padding_patch == "end":  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.patch_stride))
    
    def _compute_dataset_loss(self, data, soft_labels=None):
        """只读评估：在不更新参数的情况下计算一个数据集的对比损失均值。
        复用训练时的数据预处理、随机裁剪与损失定义，以获得可对比的指标。
        """
        if data is None or isinstance(data, (int, float)):
            return None
        assert data.ndim == 3

        # 复制并应用与训练相同的预处理
        eval_data = data
        if self.max_train_length is not None:
            sections = eval_data.shape[1] // self.max_train_length
            if sections >= 2:
                eval_data = np.concatenate(split_with_nan(eval_data, sections, axis=1), axis=0)

        temporal_missing = np.isnan(eval_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            eval_data = centerize_vary_length_series(eval_data)

        eval_data = eval_data[~np.isnan(eval_data).all(axis=2).all(axis=1)]

        dataset = custom_dataset(eval_data)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=False, drop_last=False)

        org_train_mode_net = self._net.training
        self._net.eval()

        cum_loss = 0.0
        n_iters = 0
        with torch.no_grad():
            for x, idx in loader:
                # 软标签子矩阵
                if soft_labels is None:
                    soft_labels_batch = None
                else:
                    soft_labels_batch = soft_labels[idx][:, idx]

                # 长序列随机裁剪与预处理
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset: window_offset + self.max_train_length]
                x = x.to(self.device)

                x = x.reshape(x.shape[0], x.shape[2], -1)
                if self.revin:
                    x = x.permute(0, 2, 1)
                    x = self.revin_layer(x, 'norm')
                    x = x.permute(0, 2, 1)
                if self.padding_patch == 'end':
                    x = self.padding_patch_layer(x)
                x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)  # (B, C, N, P)
                B, C, N, P = x.shape
                x = x.reshape(B * C, N, P)

                ts_l = x.size(1)
                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l + 1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))

                out1_all = self._net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
                out2_all = self._net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))

                out1 = out1_all[:, -crop_l:]
                out2 = out2_all[:, :crop_l]

                loss = hier_CL_soft(
                    out1,
                    out2,
                    soft_labels_batch,
                    lambda_=self.lambda_,
                    tau_temp=self.tau_temp,
                    temporal_unit=self.temporal_unit,
                    soft_temporal=self.soft_temporal,
                    soft_instance=self.soft_instance,
                )
                cum_loss += float(loss.item())
                n_iters += 1

        # 还原训练模式
        self._net.train(org_train_mode_net)
        if n_iters == 0:
            return None
        return cum_loss / n_iters

    def fit(self, train_data, train_labels, test_data, test_labels, soft_labels, run_dir, n_epochs=None, n_iters=None, verbose=False, valid_data=None, test_data_eval=None):
        """
        训练TS2Vec模型
        
        Args:
            train_data (numpy.ndarray): 训练数据，形状为 (n_samples, n_timestamps, n_features)
                                       缺失值应设置为NaN
            train_labels: 训练标签（用于评估，训练过程是自监督的）
            test_data: 测试数据（用于评估）
            test_labels: 测试标签（用于评估）
            soft_labels: 软标签矩阵，用于实例级软对比学习
                        形状为 (n_samples, n_samples)，表示样本间的相似性
            run_dir: 实验结果保存目录
            n_epochs: 训练的epoch数，如果指定则按epoch训练
            n_iters: 训练的迭代数，如果指定则按迭代数训练
            verbose: 是否打印训练过程信息
            
        Returns:
            loss_log: 包含每个epoch训练损失的列表
        """
        assert train_data.ndim == 3
        
        # 创建日志目录
        LOG_PATH = run_dir.replace('result','log')
        os.makedirs(LOG_PATH, exist_ok=True)
        #tb_logger = SummaryWriter(log_dir = LOG_PATH)
        
        # 设置默认训练迭代数：小数据集200次，大数据集600次
        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600
        
        # 处理过长序列：如果序列长度超过max_train_length，则分段处理
        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)

        
        # 处理变长序列：将序列居中对齐
        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)
        
        # 移除完全为NaN的样本
        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]
        
        # 创建数据加载器
        train_dataset = custom_dataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), 
                                 shuffle=True, drop_last=True)
        
        # 初始化优化器
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        
        loss_log = []
        
        # 主训练循环
        while True:
            # 检查是否达到指定的epoch数
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            cum_loss = 0        # 累积损失
            n_epoch_iters = 0   # 当前epoch的迭代数
            
            interrupted = False
            for x, idx in train_loader:
                
                # 检查是否达到指定的迭代数
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                
                # 获取当前批次的软标签矩阵
                # 如果启用软实例对比学习，需要提取对应样本间的相似性矩阵
                if soft_labels is None:
                    soft_labels_batch = None
                else:
                    # 提取当前批次样本间的相似性子矩阵
                    soft_labels_batch = soft_labels[idx][:,idx]
       
                # 如果序列过长，随机裁剪到指定长度
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset : window_offset + self.max_train_length]
                x = x.to(self.device)

                x = x.reshape(x.shape[0], x.shape[2], -1)
                # norm
                if self.revin: 
                    x = x.permute(0,2,1)
                    x = self.revin_layer(x, 'norm')
                    x = x.permute(0,2,1)
                if self.padding_patch == "end":
                    x = self.padding_patch_layer(x)
                x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride) # (B, C, N, P)
                B, C, N, P = x.shape
                x = x.reshape(B*C, N, P) # 重塑为 (B*C, N, P), N作为时间步
                
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
                
                optimizer.zero_grad()

                # ================ 编码两个子序列 ================
                # 第一个子序列：较长，包含重叠区域的后半部分
                out1_all = self._net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
                # 第二个子序列：较长，包含重叠区域的前半部分
                out2_all = self._net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
                
                # ================ 提取重叠部分进行对比 ================
                # 只对重叠的crop_l长度部分进行对比学习
                out1 = out1_all[:, -crop_l:]  # 第一个子序列的后crop_l个时间步
                out2 = out2_all[:, :crop_l]   # 第二个子序列的前crop_l个时间步
                # out形状(batch, crop_l, features) crop_l为重叠部分长度, features是每个时间步的表示向量维度
                # ================ 计算软对比损失 ================
                loss = hier_CL_soft(
                    out1,                        # 第一个子序列的编码
                    out2,                        # 第二个子序列的编码
                    soft_labels_batch,           # 软标签矩阵（用于实例级软对比）
                    lambda_=self.lambda_,        # 实例级与时间级损失的权重平衡
                    tau_temp=self.tau_temp,      # 时间级软对比的温度参数
                    temporal_unit=self.temporal_unit,  # 时间对比的最小单元
                    soft_temporal=self.soft_temporal,  # 是否使用时间级软对比
                    soft_instance=self.soft_instance   # 是否使用实例级软对比
                )
                
                # ================ 反向传播和优化 ================
                loss.backward()
                optimizer.step()
                # 更新指数移动平均模型
                self.net.update_parameters(self._net)
                    
                # 更新训练统计
                cum_loss += loss.item()
                n_epoch_iters += 1
                self.n_iters += 1
                
                # 执行迭代后回调
                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())
            
            # 如果因达到迭代数限制而中断，退出训练循环
            if interrupted:
                break
            
            # 计算并记录epoch平均损失
            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            # 额外评估：valid/test（只读，无梯度）
            val_loss = self._compute_dataset_loss(valid_data, None) if valid_data is not None else None
            tst_loss = self._compute_dataset_loss(test_data_eval, None) if test_data_eval is not None else None
            if verbose:
                msg = f"Epoch #{self.n_epochs}: train_loss={cum_loss}"
                if val_loss is not None:
                    msg += f" valid_loss={val_loss}"
                if tst_loss is not None:
                    msg += f" test_loss={tst_loss}"
                print(msg)
            
            # 注释掉的在线评估代码（可选功能）
            # 在训练过程中可以实时评估模型在分类任务上的性能
            '''
            train_repr = self.encode(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
            test_repr = self.encode(test_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
            fit_svm = eval_protocols.fit_svm
            fit_knn = eval_protocols.fit_knn

            def merge_dim01(array):
                return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])

            if train_labels.ndim == 2:
                train_repr = merge_dim01(train_repr)
                train_labels = merge_dim01(train_labels)
                test_repr = merge_dim01(test_repr)
                test_labels = merge_dim01(test_labels)

            clf = fit_knn(train_repr, train_labels)
            acc = clf.score(test_repr, test_labels)
            y_score = clf.predict_proba(test_repr)
            test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
            if test_labels_onehot.shape[1]==1:
                test_labels_onehot = np.concatenate([np.ones((test_labels_onehot.shape[0],1))-test_labels_onehot,test_labels_onehot] ,axis=1)
            
            auprc = average_precision_score(test_labels_onehot, y_score)
            
            tb_logger.add_scalar('classification/acc_knn', acc, global_step = self.n_epochs)
            tb_logger.add_scalar('classification/auprc_knn', auprc, global_step = self.n_epochs)
        
            clf = fit_svm(train_repr, train_labels)
            acc = clf.score(test_repr, test_labels)
            y_score = clf.decision_function(test_repr)
            test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
            auprc = average_precision_score(test_labels_onehot, y_score)
            tb_logger.add_scalar('classification/acc_svm', acc, global_step = self.n_epochs)
            tb_logger.add_scalar('classification/auprc_svm', auprc, global_step = self.n_epochs)
            '''
            
            # 更新epoch计数器
            self.n_epochs += 1
            # 执行epoch后回调
            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)
            
        return loss_log
    
    
    def encode(self, x, mask=None):
        '''
        input: (B*C, N, P)
        output: (B*C, N, D)
        '''
        out = self.net(x.to(self.device, non_blocking=True), mask)

        return out

    def save(self, fn):
        """
        保存模型参数到文件
        
        Args:
            fn (str): 保存文件的路径
        """
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        """
        从文件加载模型参数
        
        Args:
            fn (str): 模型文件的路径
        """
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)
    
