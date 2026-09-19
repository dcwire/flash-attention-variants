"""LLM inference through flash attention.

Tensor layout convention used everywhere in this repo (matches
``torch.nn.functional.scaled_dot_product_attention``):

    q, k, v, o : (B, H, N, D)   batch, heads, sequence, head_dim
    S, P       : (B, H, N, N)   scores / probabilities (only ever materialised by the naive path)
    L          : (B, H, N)      row-wise logsumexp, saved by flash forward for the backward pass

Softmax scale defaults to 1/sqrt(D). ``causal=True`` masks j > i.
"""

from flash_attention._todo import NotYetImplemented, todo

__all__ = ["NotYetImplemented", "todo"]
