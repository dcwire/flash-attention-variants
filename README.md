# LLM Inference through flash attention

## Building the CUDA kernels

`setup.py` compiles `csrc/` into `flash_attention/_fa_cuda*.so` with ninja. After the first
build, editing a `.cu` file recompiles only that file.

One-time setup, on the CUDA machine:

```bash
pip install ninja pytest                   # ninja is what makes the build parallel + incremental
python setup.py build_ext --inplace        # first full build
```

Every shell, or put these in your `~/.bashrc`:

```bash
# Build only for your GPU. Find it with:
#   python -c "import torch; print(torch.cuda.get_device_capability())"   # (8, 6) -> "8.6"
export TORCH_CUDA_ARCH_LIST="8.6"
# Optional: nvcc on torch headers uses ~1-2 GB RAM per job; lower this if the build swaps.
# export MAX_JOBS=4
```

Edit/test loop:

```bash
python setup.py build_ext --inplace        # rebuilds only changed files; ~2 s if nothing changed
pytest tests/test_cuda_primitives.py       # or just run pytest: it rebuilds first if csrc/ is newer
```

Tips:

- `pytest` rebuilds automatically when any file in `csrc/` is newer than the `.so`. Set
  `FA_AUTO_BUILD=0` to turn that off, and `FA_VERBOSE_BUILD=1` to see the compiler output.
- Don't include `<torch/extension.h>` in `.cu` files. It pulls in pybind11 and the Python
  headers, which roughly doubles nvcc time per file. Use `<ATen/ATen.h>`. Only `bindings.cpp`
  needs `torch/extension.h`.
- Changing `setup.py`, a compiler flag or `TORCH_CUDA_ARCH_LIST` rebuilds everything. If a build
  gets into a strange state, delete `build/` and `flash_attention/_fa_cuda*.so`.
- A new `.cu` file is picked up automatically (`setup.py` globs `csrc/`). Declare its entry point
  in `bindings.cpp`.
