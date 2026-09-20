// Task 4: fused FlashAttention-2 forward. Mirror flash_attention/reference/flash_torch.py.
//
// grid  = (ceil(N / BLOCK_Q), B * H)          one block per Q tile per (b, h)
// block = BLOCK_Q threads (or 4 warps splitting rows) ; smem: Q tile + one K tile + one V tile
// per-thread registers: O row (D floats), m, l
// loop j over K/V tiles:
//   if (causal && j * BLOCK_K > (i + 1) * BLOCK_Q - 1) break;      // early block skip
//   S = Q_i K_j^T * scale  (mask only when the tile straddles the diagonal)
//   online-softmax rescale of O, l ; O += P V_j
// epilogue: O /= l ; L = m + log(l) written as fp32 (B, H, N)
//
// Order of work: fp32 with plain FMAs and BLOCK 32 -> pass tests -> fp16 -> mma.sync -> tune.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor> flash_forward_cuda(torch::Tensor q, torch::Tensor k,
                                                            torch::Tensor v, bool causal,
                                                            double scale) {
  throw std::runtime_error("NotYetImplemented: flash_forward_cuda");
}
