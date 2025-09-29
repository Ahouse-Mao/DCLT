import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from omegaconf import OmegaConf, DictConfig

class DCLT_data_loader_v3(Dataset):
    """
    基础数据加载器：从 CSV 读取多变量时间序列，按滑动窗口切片，返回输入与预测两段：
    - seq_x: (C, seq_len)
    - seq_y: (C, pred_len)
    结合 DataLoader 的默认 collate，可得到批次 (B, C, seq_len) 与 (B, C, pred_len)。

    约定：
    - CSV 行为时间步 T，列为不同变量；若存在 'date' 列会被自动移除。
    - 数据按列进行可选标准化（z-score）。
    - 使用滑动窗口生成样本：第 i 个样本对应原始数据 [i : i+seq_len]，转置为 (C, seq_len)。

    参数:
    - 从 cfg.model 读取：seq_len / pred_len / stride
    - 从 cfg.dataset 读取：normalize / dropna / path 或 name
    """

    def __init__(
        self,
        cfg: DictConfig,
        scaler: Optional[StandardScaler] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if not hasattr(cfg, 'model'):
            raise ValueError('cfg 中缺少 model 配置')

        self.data_name = cfg.dataset.name
        self.data_path = None
        self.dtw_path = None
        self.init_path()

        # 窗口参数从 cfg.model 读取
        self.seq_len = cfg.model.seq_len
        self.pred_len = cfg.model.pred_len
        self.stride = cfg.model.stride
        # 预处理参数从 cfg.dataset 读取，带默认值
        self.normalize = cfg.dataset.normalize
        self.dropna = cfg.dataset.dropna
        self.dtype = dtype
        self._scaler = scaler  # 可能为 None

        if not os.path.isfile(self.data_path):
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
        if self.seq_len <= 0 or self.pred_len <= 0 or self.stride <= 0:
            raise ValueError('seq_len、pred_len、stride 必须为正整数')

        self._read_and_preprocess()
        self._build_indices()
    
    def init_path(self):
        root_path = os.getcwd()
        self.data_path = os.path.join(root_path, "dataset", f"{self.data_name}.csv")
        self.dtw_path = os.path.join(root_path, "DTW_matrix", f"{self.data_name}.csv")

    def _read_and_preprocess(self) -> None:
        df = pd.read_csv(self.data_path)
        if 'date' in df.columns:
            df = df.drop(columns=['date'])
        if self.dropna:
            df = df.dropna(axis=0).reset_index(drop=True)

        values = df.values.astype(np.float64)  # (T, C)
        self.T, self.C = values.shape

        if self.normalize:
            if self._scaler is None:
                self._scaler = StandardScaler()
                values = self._scaler.fit_transform(values)
            else:
                values = self._scaler.transform(values)

        self.data = values  # numpy (T, C)

    def _build_indices(self) -> None:
        # 生成所有窗口起点索引，使得窗口 [i:i+seq_len] 与未来段 [i+seq_len : i+seq_len+pred_len] 均合法
        max_start = self.T - self.seq_len - self.pred_len
        if max_start < 0:
            # 数据长度不足以构成一个窗口
            self.starts = np.array([], dtype=np.int64)
        else:
            self.starts = np.arange(0, max_start + 1, self.stride, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.starts.size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[idx])
        end = start + self.seq_len
        y_end = end + self.pred_len
        x_win = self.data[start:end]      # (seq_len, C)
        y_win = self.data[end:y_end]      # (pred_len, C)
        # 转成 (C, seq_len) / (C, pred_len)
        x = torch.as_tensor(x_win.T.copy(), dtype=self.dtype)
        y = torch.as_tensor(y_win.T.copy(), dtype=self.dtype)
        return x, y

    @property
    def scaler(self) -> Optional[StandardScaler]:
        return self._scaler


# 简单自测：当直接运行本文件时，读取项目内某个 CSV（如 dataset/weather.csv）
if __name__ == "__main__":
    # 读取 cl_conf/pretrain_cfg_v3.yaml 并构建数据集
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cl_conf', 'pretrain_cfg_v3.yaml')
    if not os.path.exists(cfg_path):
        print(f"找不到配置文件: {cfg_path}")
    else:
        cfg = OmegaConf.load(cfg_path) 
        
        ds = DCLT_data_loader_v3(cfg)
        dl = DataLoader(ds, batch_size=8, shuffle=False)
        for x, y in dl:
            print("x shape:", tuple(x.shape), "y shape:", tuple(y.shape))
            break


