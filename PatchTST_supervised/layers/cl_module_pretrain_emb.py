import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from  pathlib import Path

from cl_models.DCLT_pretrained_cl_v3_1 import LitModel


class Pretrained_emb(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.cfg = OmegaConf.load("/home/wms/South/DCLT/cl_conf/pretrain_cfg_v3.yaml")
        self.data_name = Path(configs.data_path).stem if configs.data_path else '' # 去掉.csv后缀，便于后续的索引
        self.model_path = self._init_path()
        self.load_mode = self.cfg.predict.load_mode
        self.model_state = self.cfg.predict.model_state
        self.pretrained_model = self._load_pretrained_cl_model()
        self._init_state_of_model()

    def _init_path(self,):
        model_root_path = self.cfg.predict.root_dir
        model_path = os.path.join(model_root_path, f"{self.data_name}/")
        
        return model_path

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
                model = LitModel.load_from_checkpoint(ckpt_path, cfg=self.cfg)
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
        if self.model_state == 'reason':
            self.pretrained_model.eval()
            self.pretrained_model.freeze()
        elif self.model_state == 'fine_tune':
            self.pretrained_model.train()
            # 冻结部分层
            for params in self.pretrained_model.decompose_layer.parameters():
                params.requires_grad = False
            for params in self.pretrained_model.trend_encoder.parameters():
                params.requires_grad = False
            for params in self.pretrained_model.season_encoder.parameters():
                params.requires_grad = False
            for params in self.pretrained_model.fusion.parameters():
                params.requires_grad = False
            for params in self.pretrained_model.proj_head.parameters():
                params.requires_grad = True
        else:
            raise ValueError(f"Unsupported model_state: {self.model_state}, please choose from ['reason', 'fine_tune']")

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        # 预训练模型的前向传播
        x = self.pretrained_model.predict_step(x)
        return x