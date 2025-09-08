import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, DataLoader
import torch

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