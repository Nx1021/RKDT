import torch
from launcher.utils import get_gpu_with_lowest_memory_usage

min_memory_idx = get_gpu_with_lowest_memory_usage()
device = torch.device(f"cuda:{min_memory_idx}")
torch.cuda.set_device(device)
print(f"default GPU idx: {torch.cuda.current_device()}")