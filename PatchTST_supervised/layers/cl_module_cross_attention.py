import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from layers.pretrained_cl_layers import LoadPretrainedCLModel  # 预训练模型的导入
from  pathlib import Path
from cl_models.DCLT_patchtst_pretrained_cl import LitModel
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1.纯推理（不训练）
# model.eval()  # 必须：确保正确的推理行为
# model.requires_grad_(False)  # 可选：节省内存，加速推理

# with torch.no_grad():  # 推荐：进一步优化
#     output = model(input)


# 2.特征提取（冻结预训练模型）
# pretrained_model.eval()  # 设为评估模式
# pretrained_model.requires_grad_(False)  # 冻结参数，不更新权重

# # 在上面加新的分类头进行训练
# classifier = nn.Linear(pretrained_features, num_classes)
# optimizer = torch.optim.Adam(classifier.parameters())  # 只优化分类头


# 3.微调（部分冻结）
# model.train()  # 整体保持训练模式

# # 冻结早期层
# for param in model.backbone.parameters():
#     param.requires_grad = False

# # 解冻后期层
# for param in model.classifier.parameters():
#     param.requires_grad = True

class cl_module_cross_attention(nn.Module):
    def __init__(self, configs, patch_num, patch_len, d_model, n_heads, dropout):
        super(cl_module_cross_attention, self).__init__()
        # 预训练模型与主干模型融合的参数
        self.use_cross_attention = configs.use_cross_attention
        self.cross_attention_type = configs.cross_attention_type
        self.add_pos = configs.add_pos

        # 载入预训练模型及额外数据集的参数
        self.data_name = Path(configs.data_path).stem if configs.data_path else '' # 去掉.csv后缀，便于后续的索引
        path =  "/home/wms/South/DCLT/PatchTST_supervised/pretrained_conf/pretrained_conf.yaml"
        self.yaml_cfg = OmegaConf.load(path)
        self.model_path, self.extra_dataset_path = self.__init_path()
        self.load_mode = self.yaml_cfg.load_mode
        self.model_state = self.yaml_cfg.model_state
        self.use_extra_dataset = self.yaml_cfg.use_extra_dataset

        # 加载预训练模型，配置参数及模型状态，以及额外数据集
        self.pretrained_model = self._load_pretrained_cl_model()
        self.dataset = self._init_state_of_model()
        if self.use_extra_dataset:
            self.dataset = self._init_dataset()
            self.dataloader = DataLoader(self.dataset, batch_size=self.yaml_cfg.extra_dataset_batch_size, shuffle=True)

        # 交叉注意力模块和投影层，因为位置的不同, 需要匹配不同的维度
        if self.add_pos == 'emb_x_backbone':
            self.dim_reduction = nn.Sequential(
                nn.Linear(patch_num, 1),
                nn.ReLU(),
            )
            self.cross_proj = nn.Linear(self.pretrained_model.head_args.final_out_dim, patch_len)  # 用于维度变换,对齐cross_attention的维度
            self.cross_attn = nn.MultiheadAttention(patch_len, n_heads, dropout=dropout, batch_first=True)

        elif self.add_pos == 'backbone_x_head':
            self.dim_reduction = nn.Sequential(
                nn.Linear(patch_num, 1),
                nn.ReLU(),
            )
            self.cross_proj = nn.Linear(self.pretrained_model.head_args.final_out_dim, d_model)  # 用于维度变换,对齐cross_attention的维度
            self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
    
    def __init_path(self,):
        model_root_path = self.yaml_cfg.root_dir
        model_path = os.path.join(model_root_path, f"{self.data_name}/")
        extra_dataset_root_path = self.yaml_cfg.root_dataset_path
        extra_dataset_path = os.path.join(extra_dataset_root_path, f"{self.data_name}.csv")
        return model_path, extra_dataset_path
    

    def _load_pretrained_cl_model(self,):
        """
        加载预训练的 CL 模型
        在给定目录下找时间最新的文件夹,然后把文件夹名加入model_dir并返回
        """
        model_dir = self.model_path
        try:
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"目录不存在: {model_dir}")
            
            # 获取目录下所有子文件夹及其修改时间
            subdirs = [
                (item, os.path.getmtime(os.path.join(model_dir, item)))
                for item in os.listdir(model_dir)
                if os.path.isdir(os.path.join(model_dir, item))
            ]
            
            if not subdirs:
                raise FileNotFoundError(f"在 {model_dir} 中没有找到任何子文件夹")
            
            # 按修改时间排序，获取最新的文件夹
            latest_dir = max(subdirs, key=lambda x: x[1])[0]
            model_dir = os.path.join(model_dir, latest_dir)
            
            if self.load_mode == 'ckpt':
                # 获取文件夹中对应后缀的文件
                ckpt_files = [f for f in os.listdir(model_dir) if f.endswith('.ckpt')]
                if not ckpt_files:
                    raise FileNotFoundError(f"在 {model_dir} 中没有找到 .ckpt 文件")
                
                # 如果有多个.ckpt文件，选择最新的
                ckpt_file = max(ckpt_files, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))
                ckpt_path = os.path.join(model_dir, ckpt_file)
                
                # 加载模型
                model = LitModel.load_from_checkpoint(ckpt_path)
            elif self.load_mode == 'pth':
                pass
                # state_dict = torch.load(pt_path, map_location='cuda')
                # model = LitModel()
                # model.load_state_dict(state_dict)
            elif self.load_mode == 'pt':
                pass
                # state_dict = torch.load(pt_path, map_location='cuda')
                # model = LitModel()
                # model.load_state_dict(state_dict['state_dict'], strict=False)

            return model
                
        except:
            raise FileNotFoundError(f"在 {model_dir} 中没有找到任何子文件夹")
        

    def _init_state_of_model(self,):
        """
        根据配置文件设置模型的状态 
        """
        if self.model_state == 'reason':
            self.pretrained_model.eval()
            self.pretrained_model.freeze()
        elif self.model_state == 'fine_tune': # 微调模式
            self.pretrained_model.train()  # 整体保持训练模式

            # 冻结早期层
            for param in self.pretrained_model.model.parameters():
                param.requires_grad = False

            # 解冻后期层
            for param in self.pretrained_model.proj_1.parameters():
                param.requires_grad = True
            for param in self.pretrained_model.flatten.parameters():
                param.requires_grad = True
            for param in self.pretrained_model.proj_2.parameters():
                param.requires_grad = True
        elif self.model_state == 'feat_extraction': # 特征提取
            # 特征提取（冻结预训练模型）
            # pretrained_model.eval()  # 设为评估模式
            # pretrained_model.requires_grad_(False)  # 冻结参数，不更新权重

            # # 在上面加新的分类头进行训练
            # classifier = nn.Linear(pretrained_features, num_classes)
            # optimizer = torch.optim.Adam(classifier.parameters())  # 只优化分类头
            pass
        else:
            raise ValueError(f"model_state {self.model_state} not supported")
        
    
    def _init_dataset(self,):
        """
        如果需要额外使用别的数据集的话, 可以额外加载数据集
        """
        if self.use_extra_dataset == 'DCLT_pred_dataset':
            dataset = DCLT_pred_dataset(self.extra_dataset_path)
            return dataset
        else:
            raise ValueError(f"dataset {self.use_extra_dataset} not supported")

    def forward_emb_x_backbone(self, z):
        bs, nvars, patch_len, patch_num = z.shape
        z_atten = z.reshape(bs * nvars * patch_len, patch_num)  # z: [bs*nvars*patch_len x patch_num]
        # 新版代码把外加模块集成为了一个类
        z_atten = self.dim_reduction(z_atten)  # z: [bs*nvars*patch_len x 1]
        z_atten = z_atten.squeeze(-1) # z: [bs*nvars*patch_len]
        z_atten = z_atten.reshape(bs, nvars, patch_len)  # z: [bs x nvars x patch_len]
        Q = z_atten.reshape(bs, nvars, patch_len)  # z: [bs x nvars x patch_len]

        out_info = self._forward_use_own_dataset()  # out_info: [n_vars, pretrained_dim]
        out_info = self.cross_proj(out_info)  # out_info: [n_vars, patch_len]
        K_and_V = out_info.unsqueeze(0).expand(bs, -1, -1)  # out_info: [bs, n_vars, patch_len]

        z_atten = self.cross_attn(Q, K_and_V, K_and_V)[0]  # z_atten: [bs, nvars, patch_len]
        z_atten = z_atten.reshape(bs, nvars, patch_len)  # z_atten: [bs, nvars, patch_len]
        z_atten = z_atten.unsqueeze(-1).expand(-1, -1, -1, patch_num)  # z_atten: [bs, nvars, patch_len, patch_num]

        z_fuse = z + z_atten # 残差连接

        z = z_fuse # z: [bs, nvars, patch_len, patch_num]
        return z

    def forward_backbone_x_head(self, z):
        bs, nvars, d_model, patch_num = z.shape
            
        z_atten = z.reshape(bs * nvars * d_model, patch_num)  # z: [bs*nvars*d_model x patch_num]
        z_atten = self.dim_reduction(z_atten)  # z: [bs*nvars*d_model x 1]
        z_atten = z_atten.squeeze(-1)
        Q = z_atten.reshape(bs, nvars, d_model)  # z: [bs x nvars x d_model]

        out_info = self._forward_use_own_dataset()  # out_info: [n_vars, pretrained_dim]
        out_info = self.cross_proj(out_info)  # out_info: [n_vars, d_model]
        K_and_V = out_info.unsqueeze(0).expand(bs, -1, -1)  # out_info: [bs, n_vars, d_model]

        z_atten = self.cross_attn(Q, K_and_V, K_and_V)[0]  # z_atten: [bs, nvars, d_model]
        z_atten = z_atten.reshape(bs, nvars, d_model)  # z_atten: [bs, nvars, d_model]
        z_atten = z_atten.unsqueeze(-1).expand(-1, -1, -1, patch_num)  # z_atten: [bs, nvars, d_model, patch_num]

        z_fuse = z + z_atten # 残差连接

        z = z_fuse # z: [bs, nvars, d_model, patch_num]
        return z
    
    def _forward_use_own_dataset(self,):
        output = []
        for batch in self.dataloader:
            output.append(self.pretrained_model(batch))
        output = torch.cat(output, dim=0)
        output.squeeze_(1)
        return output

class DCLT_pred_dataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self._read_data()

    def _read_data(self):
        df = pd.read_csv(self.data_path)
        # 如果存在 'date' 列，删除
        if 'date' in df.columns:
            df = df.drop(columns=['date'])
        
        # 行表示时间步，列表示不同变量
        self.data_df = df.copy()
        values = df.values  # 形状 (T, num_vars)
        self.data_length = values.shape[1]  # 数据的长度保存
        self.time_steps, self.num_vars = values.shape

        # 按列标准化（z-score）
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(values)  # 形状保持不变

        # fit_transform
        self.data = scaled  # 保存 numpy 数组

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        x = self.data[:, idx][np.newaxis, :].astype(np.float32) # 原始形状是(T, ), 变为(1, T)
        return torch.from_numpy(x)
