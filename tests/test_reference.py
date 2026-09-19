"""Task 1. Pins the oracle to two *independent* implementations in torch itself; after this
file passes, naive_attention is trusted and everything else is compared against it."""

import pytest
import torch
import torch.nn.functional as F

from flash_attention.reference.naive import NaiveMHSA, naive_attention
from tests.conftest import SHAPE_IDS, SHAPES, assert_close
from tests.generate_matrix import gen_qkv


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_naive_matches_sdpa(shape, causal, device):
    q, k, v = gen_qkv(*shape, device=device)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    assert_close(naive_attention(q, k, v, causal=causal), expected)


def test_naive_cross_length_causal_is_end_aligned(device):
    """Decode-style: 3 new queries against 10 keys; row i may see keys 0..7+i."""
    q, k, v = gen_qkv(1, 1, 3, n_k=10, head_dim=8, device=device)
    out = naive_attention(q, k, v, causal=True)
    mask = torch.ones(3, 10, dtype=torch.bool, device=device).tril(diagonal=7)
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert_close(out, expected)


def test_naive_custom_scale(device):
    q, k, v = gen_qkv(1, 2, 16, head_dim=32, device=device)
    expected = F.scaled_dot_product_attention(q, k, v, scale=0.5)
    assert_close(naive_attention(q, k, v, scale=0.5), expected)


def test_naive_softmax_is_fp32_stable(device):
    """Large logits must not overflow: the max-subtraction has to be there."""
    q, k, v = gen_qkv(1, 1, 32, head_dim=16, device=device, scale=30.0)
    out = naive_attention(q, k, v)
    assert torch.isfinite(out).all()
    assert_close(out, F.scaled_dot_product_attention(q, k, v))


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_mhsa_matches_nn_multiheadattention(causal, device):
    torch.manual_seed(0)
    E, H, B, N = 32, 4, 2, 13
    mine = NaiveMHSA(E, H, causal=causal).to(device)
    ref = torch.nn.MultiheadAttention(E, H, bias=False, batch_first=True).to(device)
    with torch.no_grad():
        ref.in_proj_weight.copy_(mine.in_proj.weight)
        ref.out_proj.weight.copy_(mine.out_proj.weight)
    x = torch.randn(B, N, E, device=device)
    mask = torch.ones(N, N, dtype=torch.bool, device=device).triu(1) if causal else None
    expected, _ = ref(x, x, x, attn_mask=mask, need_weights=False)
    assert_close(mine(x), expected)
