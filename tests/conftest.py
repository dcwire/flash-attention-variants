"""Shared fixtures + the two conventions that shape the whole harness:

1. ``NotYetImplemented`` -> xfail. ``pytest -ra`` is the progress board (flash_attention/_todo.py).
2. Tests are parametrised over (backend, device, dtype); a combination the machine can't run is
   *skipped with the reason*, never silently dropped, so the same file is valid on this Mac (cpu,
   mps) and on the CUDA box (cpu, cuda).
"""

from __future__ import annotations

import pytest
import torch

from flash_attention._todo import NotYetImplemented
from flash_attention.backends import all_backends

# ---- progress-board behaviour ------------------------------------------------------------


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and call.excinfo is not None:
        if isinstance(call.excinfo.value, NotYetImplemented):
            report.outcome = "skipped"
            report.wasxfail = f"not implemented yet: {call.excinfo.value}"


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")
    parser.addoption("--backend", action="append", default=None, help="restrict to backend name(s)")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip = pytest.mark.skip(reason="needs --runslow")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip)


# ---- devices / dtypes / tolerances ------------------------------------------------------


def available_devices() -> list[str]:
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        devs.append("mps")
    return devs


DEVICES = available_devices()
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def tolerance(dtype: torch.dtype) -> dict:
    """atol/rtol used for output comparisons. Half precision gets an order of magnitude more.

    These are deliberately loose enough that a correct fp16 kernel with fp32 accumulation
    passes, and tight enough that a missing rescale or an off-by-one mask fails by miles.
    """
    return {
        torch.float32: dict(atol=1e-4, rtol=1e-4),
        torch.float16: dict(atol=2e-2, rtol=2e-2),
        torch.bfloat16: dict(atol=5e-2, rtol=5e-2),
    }[dtype]


def assert_close(actual, expected, dtype=None, **kw):
    dtype = dtype or expected.dtype
    torch.testing.assert_close(actual.float(), expected.float(), **{**tolerance(dtype), **kw})


# ---- parametrisation helpers --------------------------------------------------------------

# Shapes chosen to catch tiling bugs: sequence lengths that are not multiples of 32/64/128,
# a single row, and one "real" size. (B, H, N, D)
# batch, nheads, seq_len, hidden_dim
SHAPES = [
    (1, 1, 1, 16),
    (1, 1, 8, 16),
    (2, 3, 64, 32),
    (1, 2, 130, 64),
    (2, 2, 257, 64),
    (1, 1, 1024, 64),
]
SHAPE_IDS = ["x".join(map(str, s)) for s in SHAPES]


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


@pytest.fixture(params=DTYPES, ids=["f32", "f16", "bf16"])
def dtype(request):
    return request.param


def _backend_params(config):
    wanted = config.getoption("--backend") if config else None
    return [
        pytest.param(b, id=b.name) for b in all_backends() if wanted is None or b.name in wanted
    ]


def pytest_generate_tests(metafunc):
    if "backend" in metafunc.fixturenames:
        metafunc.parametrize("backend", _backend_params(metafunc.config))


@pytest.fixture
def runnable(backend, device, dtype):
    """Skip (with reason) unless this backend can run on this device/dtype. Also skips the
    naive oracle at fp16 on CPU where torch's own half matmul is unsupported/slow."""
    reason = backend.unavailable(device, dtype)
    if reason:
        pytest.skip(reason)
    if device == "cpu" and dtype != torch.float32 and backend.name.startswith("torch_naive"):
        pytest.skip("fp16/bf16 on CPU is not worth testing for the oracle")
    if device == "mps" and dtype == torch.bfloat16:
        pytest.skip("bf16 on MPS is spotty across torch versions")
    return backend
