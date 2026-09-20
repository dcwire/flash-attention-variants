// Task 2: assemble the primitives. S = gemm_nt(Q, K) * scale ; causal mask ; P = softmax(S) ;
// O = gemm_nn(P, V). Three launches + a full (B, H, N, N) S in HBM - this is the memory-bound
// baseline the fused kernel is measured against. Fold the scale and the mask into the softmax
// kernel (or a tiny elementwise kernel) rather than a separate pass over S.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>

torch::Tensor gemm_nt_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_nn_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor softmax_cuda(torch::Tensor x);

torch::Tensor naive_attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                   bool causal, double scale) {
  throw std::runtime_error("NotYetImplemented: naive_attention_cuda");
}
