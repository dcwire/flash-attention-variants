import torch


# Good for kernel testing
def gen_qkv(
    batch: int,
    heads: int,
    n_q: int,
    head_dim: int,
    *,
    n_k: int | None = None,
    device,
    dtype=torch.float32,
    seed=0,
    scale=1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random (B, H, N, D) q/k/v. Callers pass either ``gen_qkv(B, H, N, D, ...)`` or
    ``gen_qkv(B, H, N_q, head_dim=D, n_k=N_k, ...)``. k and v get ``n_k`` rows (defaults to n_q).

    Values are drawn in fp32 on CPU from a seeded generator and only then cast/moved, so every
    (device, dtype) combination sees the same numbers. ``scale`` multiplies q and k only, so
    logits grow by scale**2 while v stays unit-variance.
    """
    n_k = n_q if n_k is None else n_k
    gen = torch.Generator().manual_seed(seed)

    def draw(n, s):
        x = torch.randn(batch, heads, n, head_dim, generator=gen) * s
        return x.to(device=device, dtype=dtype)

    q = draw(n_q, scale)
    k = draw(n_k, scale)
    v = draw(n_k, 1.0)
    return q, k, v
