"""Tasks 2, 4, 5 - CUDA leg, built from csrc/ by setup.py into flash_attention/_fa_cuda*.so.

``ext()`` runs ``python setup.py build_ext --inplace`` itself when that .so is missing or older
than any file in csrc/ (set FA_AUTO_BUILD=0 to turn this off). The build is incremental: ninja
recompiles only the .cu files that changed. Build output is shown when FA_VERBOSE_BUILD=1.
Set TORCH_CUDA_ARCH_LIST to your GPU's compute capability to avoid building extra archs.

The extension also exports the *building-block* kernels (softmax, gemm_nt, gemm_nn) so
tests/test_cuda_primitives.py can pin each one down before they are assembled.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from flash_attention._todo import NotYetImplemented
from flash_attention.backends import Backend

ROOT = Path(__file__).resolve().parents[2]
CSRC = ROOT / "csrc"
PKG = ROOT / "flash_attention"


def requirement() -> str | None:
    if not torch.cuda.is_available():
        return "no CUDA device"
    if shutil.which("nvcc") is None and "CUDA_HOME" not in os.environ:
        return "nvcc not on PATH and CUDA_HOME unset"
    return None


def _stale() -> bool:
    built = list(PKG.glob("_fa_cuda*.so"))
    if not built:
        return True
    so_mtime = max(p.stat().st_mtime for p in built)
    inputs = [p for p in CSRC.rglob("*") if p.is_file()] + [ROOT / "setup.py"]
    return any(p.stat().st_mtime > so_mtime for p in inputs)


def build() -> None:
    """Incremental build of csrc/ into flash_attention/ (same as `python setup.py build_ext -i`)."""
    verbose = os.environ.get("FA_VERBOSE_BUILD") == "1"
    proc = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=ROOT,
        stdout=None if verbose else subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"CUDA extension build failed:\n{proc.stdout or ''}")


@functools.lru_cache(maxsize=1)
def ext():
    if os.environ.get("FA_AUTO_BUILD", "1") != "0" and _stale():
        build()
    from flash_attention import _fa_cuda

    return _fa_cuda


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
