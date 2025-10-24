import torch; print(torch.backends.cudnn.version())

import torch
print("PyTorch 版本:", torch.__version__)
print("PyTorch 编译时使用的 CUDA 版本:", torch.version.cuda)
print("PyTorch 对应的 cuDNN 版本:", torch.backends.cudnn.version())
print("CUDA 是否可用:", torch.cuda.is_available())