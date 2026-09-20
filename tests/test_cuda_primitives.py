"""Task 2 building blocks, tested one at a time before attention.cu assembles them.
Whole file skips off-CUDA. Shapes deliberately include non-multiples of 16/32."""

import pytest
import torch

from flash_attention.backends import cuda
from tests.conftest import assert_close

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(cuda.requirement() is not None, reason=str(cuda.requirement())),
]
DTYPES = [torch.float32, torch.float16]


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f16"])
@pytest.mark.parametrize("rows,n", [(1, 1), (4, 33), (8, 1024), (2, 5000)])
def test_softmax(dtype, rows, n):
    x = torch.randn(rows, n, device="cuda", dtype=dtype) * 4
    assert_close(cuda.softmax(x), torch.softmax(x.float(), -1), dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f16"])
@pytest.mark.parametrize(
    "b,m,n,k", [(1, 1, 1, 1), (1, 16, 16, 16), (3, 33, 65, 17), (2, 128, 130, 64)]
)
def test_gemm_nt(dtype, b, m, n, k):
    a = torch.randn(b, m, k, device="cuda", dtype=dtype)
    bb = torch.randn(b, n, k, device="cuda", dtype=dtype)
    assert_close(cuda.gemm_nt(a, bb), a.float() @ bb.float().transpose(-1, -2), dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f16"])
@pytest.mark.parametrize(
    "b,m,n,k", [(1, 1, 1, 1), (1, 16, 16, 16), (3, 33, 65, 17), (2, 128, 64, 130)]
)
def test_gemm_nn(dtype, b, m, n, k):
    a = torch.randn(b, m, k, device="cuda", dtype=dtype)
    bb = torch.randn(b, k, n, device="cuda", dtype=dtype)
    assert_close(cuda.gemm_nn(a, bb), a.float() @ bb.float(), dtype=dtype)


def test_gemm_4d_batch_dims():
    a = torch.randn(2, 3, 20, 8, device="cuda")
    bb = torch.randn(2, 3, 30, 8, device="cuda")
    assert_close(cuda.gemm_nt(a, bb), a @ bb.transpose(-1, -2))
