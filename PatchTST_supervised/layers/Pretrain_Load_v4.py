import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from  pathlib import Path

from cl_models_v4 import PCLE_Model_ver2

def load_pretrained_model(configs):
    data_name = Path(configs.data_path).stem # 去掉.csv后缀，便于后续的索引
    model_state = configs.model_state
    model_dir = os.path.join("/home/wms/South/DCLT/checkpoints/", f"{data_name}/", configs.pretrain_folder)

    model, cfg = find_model(model_dir, 'pkl')
    # model = init_state_of_model(model, model_state)
    # 校验参数
    check_prams(configs, cfg)
    return model.net

def find_model(model_dir, load_mode='pkl'):
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"目录不存在: {model_dir}")

    if load_mode == 'pkl':
        # 获取文件夹中对应后缀的文件
        ckpt_files = os.path.join(model_dir, 'best.pkl')

        # 读取配置文件
        cfg_dir = os.path.join(model_dir, f"config.yaml")
        cfg = OmegaConf.load(cfg_dir)
        
        # 选择best的
        ckpt_path = os.path.join(model_dir, f"best.pkl")
        print(f"加载预训练模型: {ckpt_path}")
        if cfg.patch_cl_ver == 1:
            from cl_models_v4.PCLE_Model_ver1 import TS2Vec
            model = TS2Vec(
                **cfg
            )
        else:
            from cl_models_v4.PCLE_Model_ver2 import TS2Vec
            model = TS2Vec(cfg)

        model.load(ckpt_path)

        return model, cfg

# def init_state_of_model(model, model_state='reason'):
#     if model_state == 'reason':
#         model.net.eval()
#         for param in model.net.parameters():
#             param.requires_grad = False
#     elif model_state == 'finetune':
#         pass
#     else:
#         raise ValueError(f"未知的 model_state: {model_state}")

#     return model

def check_prams(configs, cfg):
    if configs.patch_len != cfg.patch_len:
        Warning(f"推理Patch_len与训练Patch_len不匹配: {configs.patch_len} vs {cfg.patch_len}, 但仍可运行")
    if configs.stride != cfg.patch_stride:
        Warning(f"推理Patch_stride与训练Patch_stride不匹配: {configs.stride} vs {cfg.patch_stride}, 但仍可运行")
    if configs.d_model != cfg.output_dims:
        Warning(f"embedding后的d_model维度不匹配: {configs.d_model} vs {cfg.output_dims}")
    