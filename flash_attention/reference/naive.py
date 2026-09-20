"""Task 1 - naive multi-head self-attention in PyTorch.

This is the correctness oracle for *every other backend in the repo* (see docs/DESIGN.md).
It is allowed to be slow and to materialise the full (B, H, N, N) score matrix. It must be
written in plain tensor ops (no F.scaled_dot_product_attention) so it is an independent
witness; ``tests/test_reference.py`` checks it against SDPA and nn.MultiheadAttention once.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from flash_attention._todo import todo


def naive_attention(
    q: Tensor, k: Tensor, v: Tensor, causal: bool = False, scale: float | None = None
) -> Tensor:
    """Full-matrix attention.

    Args:
        q, k, v: (B, H, N, D). k/v may have a different sequence length M than q (needed for
            decode, where N == 1 and M == cache length).
        causal: mask positions j > i (aligned to the *end*: row i of q attends to
            k[:, :, : M - N + i + 1]; for N == M that is the usual lower triangle).
        scale: softmax scale, default 1/sqrt(D).
    Returns:
        o: (B, H, N, D) in q's dtype. Do the softmax in fp32 regardless of input dtype.
    """
    # todo("naive_attention: S = q k^T * scale, mask, softmax over last dim, P v")
    # No decode yet
    # batch_size = q.size(0)
    seq_len = q.size(-2)
    device = q.device
    if scale is None:
        scale = 1 / torch.sqrt(torch.tensor(q.shape[-1]))
    S = (torch.matmul(q, k.transpose(-1, -2))) * scale

    if causal:
        # Fix for decode
        msk = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(msk, value=float("-inf"))

    S = F.softmax(S, dim=-1)

    o = S @ v

    return o


class NaiveMHSA(nn.Module):
    """Multi-head self-attention block: x -> qkv projection -> heads -> naive_attention -> out proj.

    Weight layout is chosen to match ``torch.nn.MultiheadAttention(bias=False, batch_first=True)``
    so tests can copy weights across: ``in_proj.weight`` is (3E, E) stacked [Wq; Wk; Wv].
    """

    def __init__(self, embed_dim: int, num_heads: int, causal: bool = False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim, self.num_heads, self.causal = embed_dim, num_heads, causal
        self.head_dim = embed_dim // num_heads
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, N, E) -> (B, N, E)."""
        # todo("NaiveMHSA.forward: project, split heads to (B,H,N,D), attend, merge, out_proj")
        x = self.in_proj(x)

        q, k, v = torch.split(x, x.size(-1) // 3, dim=-1)

        q = split_heads(q, self.num_heads)
        k = split_heads(k, self.num_heads)
        v = split_heads(v, self.num_heads)

        scale = k.size(-1) ** -0.5


        return self.out_proj(merge_heads(naive_attention(q, k, v, self.causal, scale)))


def split_heads(x: Tensor, num_heads: int) -> Tensor:
    """(B, N, H*D) -> (B, H, N, D)."""
    b, n, e = x.shape
    return x.view(b, n, num_heads, e // num_heads).transpose(1, 2)


def merge_heads(x: Tensor) -> Tensor:
    """(B, H, N, D) -> (B, N, H*D)."""
    b, h, n, d = x.shape
    return x.transpose(1, 2).reshape(b, n, h * d)


def causal_mask(n_q: int, n_k: int, device: torch.device) -> Tensor:
    """Boolean (n_q, n_k) mask, True where attention is *allowed*, aligned to the end."""
    i = torch.arange(n_q, device=device)[:, None]
    j = torch.arange(n_k, device=device)[None, :]
    return j <= i + (n_k - n_q)
