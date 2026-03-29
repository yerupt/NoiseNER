import torch
import sys

print(f"Python 路径: {sys.executable}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"GPU 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"当前使用的显卡: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"显存总量: {total_mem:.2f} MB")