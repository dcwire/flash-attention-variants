// Stretch: FlashAttention-2 backward (Algorithm 2). Grid over K/V tiles; each block owns dK_j,
// dV_j in registers and loops over Q tiles, recomputing S_ij and P_ij = exp(S_ij - L_i).
// dQ needs a cross-block reduction: atomicAdd into an fp32 dQ buffer, or a second kernel that
// loops the other way. Precompute Delta = rowsum(dO * O) in a small preceding kernel.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> flash_backward_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor L,
    torch::Tensor dO, bool causal, double scale) {
  throw std::runtime_error("NotYetImplemented: flash_backward_cuda");
}
