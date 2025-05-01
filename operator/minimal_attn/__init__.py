import torch
from torch import Tensor
from pathlib import Path
from . import _C
import gc

def setup_buffers(Q: Tensor):
    B = Q.size(0)
    nh = Q.size(1)
    N = Q.size(2)
    d = Q.size(3);
    SIZEOF_FLOAT = 4;
    buf1 = torch.empty(B * nh * N, dtype=torch.float32, device='cuda')
    buf2 = torch.empty(B * nh * N, dtype=torch.float32, device='cuda')
    return buf1, buf2

def print_memory():
    print(torch.cuda.memory_allocated() / 1e6, "MB allocated")
    print(torch.cuda.memory_reserved() / 1e6, "MB reserved")

def mha_forward(q: Tensor, k: Tensor, v: Tensor, use_tensor_cores: bool) -> Tensor:
    """Performs multihead attention based on the original minimal-flash-attn repo's flash.cu implementation. There are 2 cudaMalloc's in flash.cu. We want PyTorch to handle the memory, so we'll allocate it here and pass it in as the first two arguments instead."""
    print("Before:")
    print_memory()
    buf1, buf2 = setup_buffers(q)
    print("After allocating:")
    print_memory()
    out = torch.ops.minimal_attn.mha_forward(buf1, buf2, q, k, v, use_tensor_cores)
    del buf1, buf2
    gc.collect()
    torch.cuda.empty_cache()
    print("After freeing:")
    print_memory()
    return out

def improved_mha_forward(q: Tensor, k: Tensor, v: Tensor, use_tensor_cores: bool) -> Tensor:
    """Performs multihead attention based on our improved_flash.cu implementation. Again, allocate memory here internally then call the kernel function."""
    buf1, buf2 = setup_buffers(q)
    out = torch.ops.minimal_attn.improved_mha_forward(buf1, buf2, q, k, v, use_tensor_cores)
    del buf1, buf2
    gc.collect()
    torch.cuda.empty_cache()
    return out