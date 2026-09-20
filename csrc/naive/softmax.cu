// Task 2: row-wise softmax over the last dimension.
// Suggested design: one thread block per row, each thread strides over the row; two block
// reductions (max, then sum of exp(x - max)); write exp(x - max) / sum. Accumulate in fp32 even
// for fp16 input. Rows can be long (N up to 16k) so don't assume a row fits in one block's threads.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>

torch::Tensor softmax_cuda(torch::Tensor x) {
  throw std::runtime_error("NotYetImplemented: softmax_cuda");
}
