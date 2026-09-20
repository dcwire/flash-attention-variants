"""The shared forward-correctness suite. Every backend (torch tiled, CUDA naive, CUDA fused,
Triton, MLIR) runs exactly these tests against the naive oracle."""

import pytest
import torch

from flash_attention.reference.naive import naive_attention
from tests.conftest import SHAPE_IDS, SHAPES, assert_close
from tests.generate_matrix import gen_qkv


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_forward_matches_oracle(runnable, device, dtype, shape, causal):
    if runnable.name == "torch_naive":
        pytest.skip("oracle is tested in test_reference.py")
    q, k, v = gen_qkv(*shape, device=device, dtype=dtype)
    expected = naive_attention(q.float(), k.float(), v.float(), causal=causal)
    out = runnable(q, k, v, causal=causal)
    assert out.shape == q.shape and out.dtype == q.dtype
    assert_close(out, expected, dtype=dtype)


@pytest.mark.parametrize("n_q,n_k", [(1, 64), (5, 64), (64, 200), (37, 37)])
def test_forward_cross_length(runnable, device, dtype, n_q, n_k):
    if not runnable.supports_cross_len:
        pytest.skip(f"{runnable.name} requires N_q == N_k")
    q, k, v = gen_qkv(1, 2, n_q, n_k=n_k, head_dim=32, device=device, dtype=dtype)
    expected = naive_attention(q.float(), k.float(), v.float(), causal=True)
    assert_close(runnable(q, k, v, causal=True), expected, dtype=dtype)


def test_forward_large_logits_stable(runnable, device, dtype):
    """Catches a missing running-max rescale: exp() overflows in fp16 without it."""
    q, k, v = gen_qkv(1, 1, 256, head_dim=64, device=device, dtype=dtype, scale=8.0)
    out = runnable(q, k, v)
    assert torch.isfinite(out).all()
    assert_close(out, naive_attention(q.float(), k.float(), v.float()), dtype=dtype)


def test_forward_custom_scale(runnable, device, dtype):
    q, k, v = gen_qkv(1, 1, 64, head_dim=64, device=device, dtype=dtype)
    expected = naive_attention(q.float(), k.float(), v.float(), scale=0.03)
    assert_close(runnable(q, k, v, scale=0.03), expected, dtype=dtype)


def test_forward_does_not_mutate_inputs(runnable, device, dtype):
    q, k, v = gen_qkv(1, 1, 64, head_dim=64, device=device, dtype=dtype)
    q0, k0, v0 = q.clone(), k.clone(), v.clone()
    runnable(q, k, v, causal=True)
    assert torch.equal(q, q0) and torch.equal(k, k0) and torch.equal(v, v0)
