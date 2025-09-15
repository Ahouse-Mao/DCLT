import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, DataLoader
import torch

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)

from cl_models.DCLT_patchtst_pretrained_cl import LitModel

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

class LoadPretrainedCLModel():
    def __init__(self, yaml_cfg):
        super().__init__()
        self.cfg = yaml_cfg

        self.model_dir = yaml_cfg.model_dir
        self.load_mode = yaml_cfg.load_mode
        self.model_state = yaml_cfg.model_state
        self.use_extra_dataset = yaml_cfg.use_extra_dataset
        self.dataset_path = yaml_cfg.dataset_path

        self.pretrained_model = self._load_pretrained_cl_model()
        self.dataset = self._init_state_of_model()
        if self.use_extra_dataset:
            self.dataset = self._init_dataset()
            self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)

    def _load_pretrained_cl_model(self,):
        """
        加载预训练的 CL 模型
        在给定目录下找时间最新的文件夹，然后把文件夹名加入model_dir并返回
        """
        model_dir = self.model_dir
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
            dataset = DCLT_pred_dataset(self.dataset_path)
            return dataset
        else:
            raise ValueError(f"dataset {self.use_extra_dataset} not supported")
        
    def forward(self, x):
        return self.pretrained_model(x)
    
    def forward_use_own_dataset(self,):
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


from omegaconf import OmegaConf

if __name__ == "__main__":
    # yaml_cfg = OmegaConf.load("../pretrained_conf/pretrained_conf.yaml")
    path = os.path.join(os.getcwd(), "PatchTST_supervised/pretrained_conf/pretrained_conf.yaml")
    yaml_cfg = OmegaConf.load(path)

    model_loader = LoadPretrainedCLModel(yaml_cfg)
    output = model_loader.forward_use_own_dataset()
    print(output.shape)  # (num_vars, final_out_dim)