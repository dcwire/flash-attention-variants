"""Online softmax in 1D - derive this on paper first (docs/notes/online_softmax.md).

The whole of FlashAttention rests on one identity. For a vector x split into blocks
x = [x_1 | x_2 | ...], keep per-block (m_b = max x_b, l_b = sum exp(x_b - m_b)). Then for the
running state (m, l) after seeing blocks 1..t and a new block with (m', l'):

    m_new = max(m, m')
    l_new = l * exp(m - m_new) + l' * exp(m' - m_new)

and softmax(x)_i = exp(x_i - m_final) / l_final. In attention the same rescale factor
exp(m - m_new) is applied to the accumulated output rows O (Algorithm 1, lines 9-10 of the
FlashAttention-2 paper) - that is the "rescale accumulator per block" step of the CUDA leg.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from flash_attention._todo import todo


@dataclass
class SoftmaxState:
    """Running statistics for one softmax row (or a batch of rows along dim 0)."""

    m: Tensor  # running max
    l: Tensor  # running sum of exp(x - m)

    @staticmethod
    def empty(like: Tensor) -> SoftmaxState:
        return SoftmaxState(m=torch.full_like(like, float("-inf")), l=torch.zeros_like(like))

    def update(self, block: Tensor) -> SoftmaxState:
        """Fold a new block of scores (…, block_size) into the state; returns the new state."""
        todo("SoftmaxState.update: block max, rescale l by exp(m - m_new), add block sum")

    @staticmethod
    def merge(a: SoftmaxState, b: SoftmaxState) -> SoftmaxState:
        """Combine two partial states (this is also what split-KV decode reduces with)."""
        todo("SoftmaxState.merge")


def online_softmax(x: Tensor, block_size: int) -> Tensor:
    """softmax over the last dim of ``x`` using only ``block_size`` scores at a time.

    Two passes are allowed (one to get (m, l), one to normalise) - the point is that the
    first pass never needs more than one block in memory and never sees the global max.
    """
    todo("online_softmax")
