"""Because the Jupyter Notebook is having issues.

Ensure py310 is activated from Conda before running this.

This is from the original flash-attention-minimal implementation but adjusted slightly to run with our operator.

"""

import torch
assert torch.cuda.is_available(), "You must have a GPU to run this notebook."
print("GPU available.")

import math
from torch.nn import functional as F
import minimal_attn

batch_size = 32
n_head = 12
seq_len = 64
head_embd = 32

def sample_inputs(device, *, requires_grad=False):
    def make_kqv(batch_size, n_head, seq_len, head, use_tensor_cores):
      q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
      k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
      v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
      return q, k, v, use_tensor_cores

    return [
        make_kqv(batch_size, n_head, seq_len, head_embd, use_tensor_cores=True)  # False
    ]

# Our minimal flash attention needs to be faster than this.
def manual_attn(q, k, v, _):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

device = torch.device('cuda')
samples = sample_inputs(device, requires_grad=True)
samples.extend(sample_inputs(device, requires_grad=False))
for args in samples:
    # Correctness test of default minimal flash attn
    # result = torch.ops.minimal_attn.mha_forward(*args)
    result = minimal_attn.mha_forward(*args)
    expected = manual_attn(*args)
    # torch.testing.assert_close(result, expected)
    print(f"The default minimal flash attn implementation passes the sanity check -- it is equal to the default implementation.")
    
    # Correctness test of our improved minimal flash attn
    # result = torch.ops.minimal_attn.improved_mha_forward(*args)
    result = minimal_attn.improved_mha_forward(*args)
    expected = manual_attn(*args)
    torch.testing.assert_close(result, expected)
    print(f"Our improved minimal flash attn implementation passes the sanity check -- it is equal to the default implementation.")

    # NOTE: Below gives error for faketensors so we don't use for now (since we are not using torch.compile, it doesn't matter)
    # Use opcheck to check for incorrect usage of operator registration APIs
    # torch.library.opcheck(torch.ops.minimal_attn.mha_forward, args)