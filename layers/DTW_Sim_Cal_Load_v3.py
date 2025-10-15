import os
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from hydra.utils import get_original_cwd

class DTW_Sim:
    """DTW 结果计算与缓存工具。"""

    def __init__(self, dataset_name, patch_len):
        self.dataset_name = dataset_name
        self.patch_len = str(patch_len)
        self.root_path = get_original_cwd()
        self.path = os.path.join(self.root_path, "dtw_cache", self.dataset_name, self.patch_len)
        os.makedirs(self.path, exist_ok=True)

    def get_sim_mat(self, batch_data, batch_idx):
        """读取指定 batch 的 DTW 缓存，若不存在则现算现存。"""
        file_path = os.path.join(self.path, f"dtw_{batch_idx}.cvs")
        if not os.path.exists(file_path):
            distance = self.compute_and_save(batch_data, batch_idx)

        mat = np.loadtxt(file_path, delimiter=",", ndmin=2)
        col = mat.shape[1]
        side = int(round(col ** 0.5))
        if side * side != col:
            raise ValueError(f"缓存文件 {file_path} 的列数 {col} 无法还原为方阵。")
        return mat.reshape(mat.shape[0], side, side)

    def compute_and_save(self, batch_data, batch_idx):
        """计算一个 batch 的 DTW 矩阵并保存到缓存目录。"""
        distances = self.batch_variable_dtw(batch_data)
        distances = np.asarray(distances)
        B, C, _ = distances.shape
        flattened = distances.reshape(B, C * C)

        file_path = os.path.join(self.path, f"dtw_{batch_idx}.cvs")
        np.savetxt(file_path, flattened, delimiter=",")
        return distances

    # 运行有一次(32,321,336)的输入大概需要60秒
    def batch_variable_dtw(data):
        # data(B, C, L)，B 为批次大小，C 为变量数，L 为序列长度。
        # 返回(B, C, C)的距离矩阵
        # if torch is not None and isinstance(data, torch.Tensor):
        # 	data = data.detach().cpu().numpy()
        B, C, _ = data.shape
        dtw_mat = np.zeros((B, C, C), dtype=np.float32)
        for b in range(B):
            sample = data[b]
            for i in range(C - 1):
                for j in range(i + 1, C):
                    distance, _ = fastdtw(sample[i].unsqueeze(0), sample[j].unsqueeze(0), dist=euclidean)
                    dtw_mat[b, i, j] = distance
                    dtw_mat[b, j, i] = distance
        return dtw_mat