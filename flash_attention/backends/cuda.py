"""Tasks 2, 4, 5 - CUDA leg, loaded as a JIT-compiled torch extension.

``torch.utils.cpp_extension.load`` compiles csrc/ on first use into
~/.cache/torch_extensions (override with TORCH_EXTENSIONS_DIR). No setup.py needed; a change to
any .cu triggers a rebuild on the next import. Build output goes to stderr when
FA_VERBOSE_BUILD=1.

The extension also exports the *building-block* kernels (softmax, gemm_nt, gemm_nn) so
tests/test_cuda_primitives.py can pin each one down before they are assembled.
"""

from __future__ import annotations

import functools
import os
import shutil
from pathlib import Path

import torch

from flash_attention._todo import NotYetImplemented
from flash_attention.backends import Backend

CSRC = Path(__file__).resolve().parents[2] / "csrc"
SOURCES = [
    CSRC / "bindings.cpp",
    CSRC / "naive" / "softmax.cu",
    CSRC / "naive" / "gemm_nt.cu",
    CSRC / "naive" / "gemm_nn.cu",
    CSRC / "naive" / "attention.cu",
    CSRC / "flash" / "flash_fwd.cu",
    CSRC / "flash" / "flash_bwd.cu",
    CSRC / "flash" / "flash_decode.cu",
]


def requirement() -> str | None:
    if not torch.cuda.is_available():
        return "no CUDA device"
    if shutil.which("nvcc") is None and "CUDA_HOME" not in os.environ:
        return "nvcc not on PATH and CUDA_HOME unset"
    return None


@functools.lru_cache(maxsize=1)
def ext():
    from torch.utils.cpp_extension import load

    return load(
        name="fa_cuda",
        sources=[str(s) for s in SOURCES],
        extra_cuda_cflags=["-O3", "-lineinfo", "--expt-relaxed-constexpr"],
        extra_cflags=["-O3"],
        verbose=os.environ.get("FA_VERBOSE_BUILD") == "1",
    )


def _unwrap(call):
    """Turn the C++ stubs' 'NotYetImplemented' throw into the Python marker (-> xfail)."""
    try:
        return call()
    except RuntimeError as e:  # pybind maps std::runtime_error to RuntimeError
        if "NotYetImplemented" in str(e):
            raise NotYetImplemented(str(e)) from None
        raise


# ---- primitives (Task 2) -------------------------------------------------------------------
def softmax(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax over the last dim of a contiguous (..., N) fp32/fp16 tensor."""
    return _unwrap(lambda: ext().softmax(x))


def gemm_nt(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(…, M, K) x (…, N, K)^T -> (…, M, N). 'NT' = B is used transposed (this is Q K^T)."""
    return _unwrap(lambda: ext().gemm_nt(a, b))


def gemm_nn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(…, M, K) x (…, K, N) -> (…, M, N). (this is P V)."""
    return _unwrap(lambda: ext().gemm_nn(a, b))


# ---- assembled attention ops --------------------------------------------------------------
def naive_attention(q, k, v, causal=False, scale=None):
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    return _unwrap(lambda: ext().naive_attention(q, k, v, causal, scale))


class _FlashFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, scale):
        o, L = _unwrap(lambda: ext().flash_forward(q, k, v, causal, scale))
        ctx.save_for_backward(q, k, v, o, L)
        ctx.causal, ctx.scale = causal, scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, L = ctx.saved_tensors
        dq, dk, dv = _unwrap(
            lambda: ext().flash_backward(q, k, v, o, L, do.contiguous(), ctx.causal, ctx.scale)
        )
        return dq, dk, dv, None, None


def flash_attention(q, k, v, causal=False, scale=None):
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    return _FlashFn.apply(q, k, v, causal, scale)


def decode_attention(q, k_cache, v_cache, seq_len: int, num_splits: int = 4):
    return _unwrap(lambda: ext().flash_decode(q, k_cache, v_cache, seq_len, num_splits))


NAIVE_BACKEND = Backend(
    name="cuda_naive",
    attention=naive_attention,
    devices=("cuda",),
    dtypes=(torch.float32, torch.float16),
    requirement=requirement,
)
FLASH_BACKEND = Backend(
    name="cuda_flash",
    attention=flash_attention,
    devices=("cuda",),
    dtypes=(torch.float16, torch.bfloat16, torch.float32),
    supports_backward=True,
    requirement=requirement,
)
