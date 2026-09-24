import torch

# Good for kernel testing
def gen_qkv(*shape, n_k=10, n_q=10, head_dim=10, device, dtype=torch.float, seed=0, scale=1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    # print(shape)
    def draw(n, scale):
        x = torch.randn(*shape, head_dim, generator=gen) * scale
        return x.to(device=device, dtype=dtype)

    q = draw(n_q, scale)
    k = draw(n_k, scale)
    v = draw(n_k, 1.0)
    return q, k, v
