import os
import numpy as np
import pickle
import torch
import random
from datetime import datetime
from torch.utils.data import Dataset

def pkl_save(name, var):
    with open(name, 'wb') as f:
        pickle.dump(var, f)

def pkl_load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
    
def torch_pad_nan(arr, left=0, right=0, dim=0):
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
    return arr
    
def pad_nan_to_target(array, target_length, axis=0, both_side=True):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    
    if both_side:
        left_pad_size = pad_size // 2
        right_pad_size = pad_size - left_pad_size
    else:
        left_pad_size = 0
        right_pad_size = pad_size
    
    # 获取左边界和右边界的值用于填充
    left_slice = [slice(None)] * array.ndim
    left_slice[axis] = slice(0, 1)
    left_value = array[tuple(left_slice)]
    
    right_slice = [slice(None)] * array.ndim
    right_slice[axis] = slice(-1, None)
    right_value = array[tuple(right_slice)]
    
    # 创建左右填充数组
    parts = []
    if left_pad_size > 0:
        left_pad = np.repeat(left_value, left_pad_size, axis=axis)
        parts.append(left_pad)
    
    parts.append(array)
    
    if right_pad_size > 0:
        right_pad = np.repeat(right_value, right_pad_size, axis=axis)
        parts.append(right_pad)
    
    return np.concatenate(parts, axis=axis)

def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs

def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]

def data_dropout(arr, p):
    B, T = arr.shape[0], arr.shape[1]
    mask = np.full(B*T, False, dtype=np.bool)
    ele_sel = np.random.choice(
        B*T,
        size=int(B*T*p),
        replace=False
    )
    mask[ele_sel] = True
    res = arr.copy()
    res[mask.reshape(B, T)] = np.nan
    return res

def name_with_datetime(prefix='default'):
    now = datetime.now()
    return prefix + '_' + now.strftime("%Y%m%d_%H%M%S")

def init_dl_program(
    device_name,
    seed=None,
    use_cudnn=True,
    deterministic=False,
    benchmark=False,
    use_tf32=False,
    max_threads=None
):
    import torch
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)
        
    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)
        
    if isinstance(device_name, (str, int)):
        device_name = [device_name]
    
    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        
    return devices if len(devices) > 1 else devices[0]

class custom_dataset(Dataset): 
  def __init__(self, X):
    self.X = X

  def __len__(self): 
    return len(self.X)

  def __getitem__(self, idx): 
    X = torch.FloatTensor(self.X[idx])
    return X, idx


class sliding_window_dataset(Dataset):
    """
    滑动窗口数据集，用于将长序列数据切分为固定长度的窗口
    支持重复区间的切分
    
    输入数据格式: (1, L, C) 或 (L, C)
    输出数据格式: (B, C, L) 其中 B 是窗口数量，L 是窗口长度
    
    Args:
        data: numpy array 或 torch tensor, 形状为 (1, L, C) 或 (L, C)
        window_length: 每个窗口的长度
        stride: 滑动窗口的步长，默认等于 window_length（无重叠）
    """
    def __init__(self, data, window_length, stride=None):
        # 处理输入数据格式
        if data.ndim == 3 and data.shape[0] == 1:
            # (1, L, C) -> (L, C)
            data = data.squeeze(0)
        elif data.ndim != 2:
            raise ValueError(f"Expected data shape (1, L, C) or (L, C), got {data.shape}")
        
        self.data = data  # (L, C)
        self.window_length = window_length
        self.stride = stride if stride is not None else window_length
        
        # 计算可以生成多少个窗口
        seq_len = data.shape[0]
        self.num_windows = (seq_len - window_length) // self.stride + 1
        
        if self.num_windows <= 0:
            raise ValueError(f"Sequence length {seq_len} is too short for window length {window_length}")
    
    def __len__(self):
        return self.num_windows
    
    def __getitem__(self, idx):
        """
        返回一个窗口的数据
        
        Returns:
            X: torch.FloatTensor, 形状为 (C, L)
            idx: int, 窗口索引
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_length
        
        # 提取窗口数据: (L, C) -> (window_length, C)
        window_data = self.data[start_idx:end_idx, :]
        
        # 转置为 (C, L) 
        X = window_data.T  # (C, window_length)
        
        return X, idx



"""
This module is used to show the shape of params when debugging.
"""

import torch

original_repr = torch.Tensor.__repr__
def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}, {self.device}, {self.dtype}}} {original_repr(self)}'

def show_shape():
    torch.Tensor.__repr__ = custom_repr

class custom_dataset(Dataset): 
  def __init__(self, X):
    self.X = X

  def __len__(self): 
    return self.X.shape[0]

  def __getitem__(self, idx): 
    X = torch.FloatTensor(self.X[idx])
    return X, idx
  
  # ...existing code...
def get_a_vector(A, indx, num_elem):
    """
    Torch-only 实现：
    - A: torch.Tensor, shape (B, T, ...)
    - indx: array-like torch tensor, shape (B,)
    - num_elem: int, 每行要取的连续元素个数
    返回: torch.Tensor, shape (B, num_elem, ...)
    会在索引越界时抛 IndexError。
    """
    if not torch.is_tensor(A):
        raise TypeError("A must be a torch.Tensor")

    B = A.shape[0]
    T = A.shape[1]
    indx=indx.to(A.device)

    # 边界检查：确保所有需要的索引都在 [0, T-1] 内
    min_idx = int(indx.min().item())
    max_needed = int((indx + (num_elem - 1)).max().item())
    if min_idx < 0 or max_needed >= T:
        raise IndexError(f"Requested windows exceed time dimension: "
                         f"A.shape[1]={T}, required max index={max_needed}, min index={min_idx}")

    # 构造索引并选取
    ar = torch.arange(num_elem, device=A.device).unsqueeze(0)     # 1 x num_elem
    all_indx = indx.unsqueeze(1) + ar                          # B x num_elem
    rows = torch.arange(B, device=A.device).unsqueeze(1)         # B x 1 -> broadcast
    return A[rows, all_indx]
# ...existing code...