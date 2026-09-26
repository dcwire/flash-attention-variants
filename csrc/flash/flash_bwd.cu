// Stretch: FlashAttention-2 backward (Algorithm 2). Grid over K/V tiles; each block owns dK_j,
// dV_j in registers and loops over Q tiles, recomputing S_ij and P_ij = exp(S_ij - L_i).
// dQ needs a cross-block reduction: atomicAdd into an fp32 dQ buffer, or a second kernel that
// loops the other way. Precompute Delta = rowsum(dO * O) in a small preceding kernel.
#include <ATen/ATen.h>  // not torch/extension.h: only bindings.cpp needs pybind
#include <cuda_runtime.h>
#include <stdexcept>
#include <tuple>

std::tuple<at::Tensor, at::Tensor, at::Tensor> flash_backward_cuda(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o, at::Tensor L,
    at::Tensor dO, bool causal, double scale) {
  throw std::runtime_error("NotYetImplemented: flash_backward_cuda");
}
