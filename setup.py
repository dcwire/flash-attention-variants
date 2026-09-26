"""Builds the CUDA leg (csrc/) as flash_attention/_fa_cuda*.so. Metadata lives in pyproject.toml.

Day-to-day loop (incremental: ninja recompiles only the .cu files you touched):

    export TORCH_CUDA_ARCH_LIST="8.6"         # your GPU only; see README
    python setup.py build_ext --inplace

flash_attention.backends.cuda also runs that command itself when the .so is missing or older than
any file in csrc/, so plain `pytest` works too.

Why not torch.utils.cpp_extension.load (JIT)? It renames the module fa_cuda_v1, _v2, ... on
every source change and bakes the name into every compile command via -DTORCH_EXTENSION_NAME,
so ninja rebuilt all translation units on each edit. Here the name is fixed.
"""

from pathlib import Path

from setuptools import setup

ROOT = Path(__file__).resolve().parent
SOURCES = sorted(
    str(p.relative_to(ROOT)) for p in (ROOT / "csrc").rglob("*") if p.suffix in (".cpp", ".cu")
)


def ext_modules():
    try:
        from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
    except ImportError:
        print("setup.py: torch not importable (use --no-build-isolation); skipping CUDA extension")
        return [], {}
    if CUDA_HOME is None:
        print("setup.py: nvcc/CUDA_HOME not found; skipping CUDA extension")
        return [], {}
    ext = CUDAExtension(
        name="flash_attention._fa_cuda",
        sources=SOURCES,
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", "-lineinfo", "--expt-relaxed-constexpr"],
        },
    )
    return [ext], {"build_ext": BuildExtension}


modules, cmdclass = ext_modules()
setup(ext_modules=modules, cmdclass=cmdclass)
