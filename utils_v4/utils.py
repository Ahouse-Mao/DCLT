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
    
def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size//2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs

def take_per_row(A, indx, num_elem):
    """
    等价功能（更稳健的纯 Torch 实现）:
    - A: Tensor, 形状 (B, N, ...)，可在 CPU 或 CUDA 上
    - indx: (B,) 的起始下标（可为 numpy/tensor/list），每行从 indx[b] 开始取连续 num_elem 个元素
    - num_elem: int

    返回: (B, num_elem, ...) 的切片结果，带越界断言，索引 dtype/device 一致。
    """
    if not isinstance(num_elem, int):
        num_elem = int(num_elem)

    device = A.device
    B, N = A.shape[0], A.shape[1]

    # 将 indx 转为同设备 long 张量，并校验形状
    idx = torch.as_tensor(indx, device=device, dtype=torch.long)
    if idx.ndim != 1:
        idx = idx.view(-1)
    assert idx.shape[0] == B, f"take_per_row: idx.shape[0] ({idx.shape[0]}) 必须等于 batch 大小 B ({B})"

    # 严格越界检查（更早暴露问题，而不是触发不明确的 CUDA 错误）
    if torch.any(idx < 0) or torch.any(idx + num_elem > N):
        raise IndexError(
            f"take_per_row: 索引越界: 需要满足 0 <= idx 且 idx+num_elem <= N，"
            f" 其中 idx.min={int(idx.min().item())}, idx.max={int(idx.max().item())}, "
            f"num_elem={num_elem}, N={N}"
        )

    # 构造 (B, num_elem) 的位置索引，并进行高级索引
    offs = torch.arange(num_elem, device=device, dtype=torch.long)  # (num_elem,)
    all_idx = idx.unsqueeze(1) + offs.unsqueeze(0)                  # (B, num_elem)
    batch_idx = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1)  # (B, 1)

    return A[batch_idx, all_idx]

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



"""
This module is used to show the shape of params when debugging.
"""

import torch

original_repr = torch.Tensor.__repr__
def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}, {self.device}, {self.dtype}}} {original_repr(self)}'

def show_shape():
    torch.Tensor.__repr__ = custom_repr