import torch
from torch import Tensor

__all__ = ["mha_forward"]

torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);

def mha_forward(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Performs multihead attention based on our flash.cu implementation."""
    return torch.ops.minimal_attn.mha_forward.default(q, k, v)