// Task 2: C[b, m, n] = sum_k A[b, m, k] * B[b, n, k]      (A @ B^T, batched over leading dims)
// Used for S = Q K^T. Start with a 16x16 shared-memory tiled GEMM with bounds checks (N is not
// a multiple of the tile in tests). The stretch goal swaps the inner product for mma.sync
// (m16n8k16 fp16 fragments) - keep the tiling so only the inner loop changes.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>

torch::Tensor gemm_nt_cuda(torch::Tensor a, torch::Tensor b) {
  throw std::runtime_error("NotYetImplemented: gemm_nt_cuda");
}
