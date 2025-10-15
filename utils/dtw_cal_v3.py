from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
import torch
import time

# 运行有一次(32,321,336)的输入大概需要60秒
def batch_variable_dtw(data):
	st_time = time.time()
    # data(B, C, L)，B 为批次大小，C 为变量数，L 为序列长度。
	# 返回(B, C, C)的距离矩阵
	# if torch is not None and isinstance(data, torch.Tensor):
	# 	data = data.detach().cpu().numpy()
	B, C, _ = data.shape
	out = np.zeros((B, C, C), dtype=np.float32)

	for b in range(B):
		sample = data[b]
		for i in range(C - 1):
			for j in range(i + 1, C):
				distance, _ = fastdtw(sample[i].unsqueeze(0), sample[j].unsqueeze(0), dist=euclidean)
				out[b, i, j] = distance
				out[b, j, i] = distance
	ed_time = time.time()
	print(f"DTW 计算时间: {ed_time - st_time} 秒")
	return out

if __name__ == "__main__":
    x = torch.randn(32, 321, 336)
    y = batch_variable_dtw(x)
    print(y)
