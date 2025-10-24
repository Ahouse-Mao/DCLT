import torch; print(torch.backends.cudnn.version())

import torch
print("PyTorch 版本:", torch.__version__)
print("PyTorch 编译时使用的 CUDA 版本:", torch.version.cuda)
print("PyTorch 对应的 cuDNN 版本:", torch.backends.cudnn.version())
print("CUDA 是否可用:", torch.cuda.is_available())
print(torch.version.cuda)

# 1. 检查CUDA是否可用
print("CUDA available:", torch.cuda.is_available())

# 2. 查看PyTorch关联的CUDA版本
if torch.cuda.is_available():
    print("PyTorch CUDA version:", torch.version.cuda)
    # 3. 尝试获取运行时CUDA工具包路径（关键步骤）
    try:
        print("CUDA_HOME:", torch.utils.cpp_extension.CUDA_HOME)
    except:
        print("Could not determine CUDA_HOME.")