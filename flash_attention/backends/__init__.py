"""Backend registry - the seam that makes one test suite serve every leg of the project.

A backend is anything with the signature

    attention(q, k, v, causal: bool = False, scale: float | None = None) -> o

on (B, H, N, D) tensors. Tests are parametrised over ``all_backends()`` and skip a backend
(not fail) when ``backend.unavailable(device)`` returns a reason. Adding a leg == adding one
module here; it inherits every correctness test, the backward test if it opts in, and the
benchmark.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor

AttentionFn = Callable[..., Tensor]


@dataclass(frozen=True)
class Backend:
    name: str
    attention: AttentionFn
    devices: tuple[str, ...] = ("cpu", "mps", "cuda")
    dtypes: tuple[torch.dtype, ...] = (torch.float32, torch.float16, torch.bfloat16)
    supports_backward: bool = False
    supports_cross_len: bool = True  # N_q != N_k (decode-style shapes)
    # returns a human-readable reason the backend can't run right now, or None
    requirement: Callable[[], str | None] = field(default=lambda: None)
    # extra per-call kwargs (tile sizes etc.) - benchmarks sweep these
    extra_kwargs: dict = field(default_factory=dict)

    def unavailable(self, device: str, dtype: torch.dtype | None = None) -> str | None:
        if device not in self.devices:
            return f"{self.name} does not run on {device}"
        if dtype is not None and dtype not in self.dtypes:
            return f"{self.name} does not support {dtype}"
        return self.requirement()

    def __call__(self, q, k, v, causal=False, scale=None):
        return self.attention(q, k, v, causal=causal, scale=scale, **self.extra_kwargs)


def all_backends() -> list[Backend]:
    # Imported lazily so a broken optional leg (no triton, no nvcc) never breaks collection.
    # from flash_attention.backends import cuda, mlir, torch_flash, torch_naive, triton_fa
    from flash_attention.backends import torch_naive, cuda

    return [
        torch_naive.BACKEND,
        cuda.NAIVE_BACKEND
    ]
    # return [
    #     torch_naive.BACKEND,
    #     torch_flash.BACKEND,
    #     cuda.NAIVE_BACKEND,
    #     cuda.FLASH_BACKEND,
    #     triton_fa.BACKEND,
    #     mlir.BACKEND,
    # ]


def get_backend(name: str) -> Backend:
    for b in all_backends():
        if b.name == name:
            return b
    raise KeyError(f"unknown backend {name!r}; known: {[b.name for b in all_backends()]}")
