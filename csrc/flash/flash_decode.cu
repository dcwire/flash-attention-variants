// Task 5: decode kernel. q is (B, H, 1, D); k/v caches are (B, H, max_seq, D) with seq_len valid.
// grid = (num_splits, B * H): each block reduces cache[split_start : split_end] into a partial
// (O_s, m_s, l_s) in a scratch buffer, then a merge kernel (or the last block, via a counter)
// combines the partials with the online-softmax merge rule and writes O.
// Only the reduction structure differs from flash_fwd - no Q tiling, no causal mask.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>

torch::Tensor flash_decode_cuda(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
                                int64_t seq_len, int64_t num_splits) {
  throw std::runtime_error("NotYetImplemented: flash_decode_cuda");
}
