import torch
from torchvision.models import *
import pynvml
from contextlib import contextmanager

@contextmanager
def track_gpu_memory():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    try:
        yield
    finally:
        print(f"Max GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e6:.2f} MB")
        print(f"Current GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")

def detailed_memory_info():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"Total GPU memory: {info.total / 1e6:.2f} MB")
    print(f"Free GPU memory: {info.free / 1e6:.2f} MB")
    print(f"Used GPU memory: {info.used / 1e6:.2f} MB")
    pynvml.nvmlShutdown()

# detailed_memory_info()

