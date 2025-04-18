import torch
from torch import Tensor

__all__ = ["mha_forward"]

torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool use_tensor_cores);

def mha_forward(q: Tensor, k: Tensor, v: Tensor, bool: use_tensor_cores) -> Tensor:
    """Performs multihead attention based on our flash.cu implementation."""
    return torch.ops.minimal_attn.mha_forward(q, k, v, use_tensor_cores)

# NOTE: The below is nice to have but unnecessary for now since we are not using torch.compile.
# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
# @torch.library.register_fake("minimal_attn::mha_forward")
# def _(q, k, v, use_tensor_cores):
#     torch._check(q.shape == k.shape == v.shape)
#     torch._check(q.dtype == torch.float)
#     torch._check(k.dtype == torch.float)
#     torch._check(v.dtype == torch.float)    
#     torch._check(q.device == k.device == v.device)
#     return torch.empty_like(q)
