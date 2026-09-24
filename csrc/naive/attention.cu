// Task 2: assemble the primitives. S = gemm_nt(Q, K) * scale ; causal mask ; P = softmax(S) ;
// O = gemm_nn(P, V). Three launches + a full (B, H, N, N) S in HBM - this is the memory-bound
// baseline the fused kernel is measured against. Fold the scale and the mask into the softmax
// kernel (or a tiny elementwise kernel) rather than a separate pass over S.
#include <cstdio>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>

// This generates 3 extra tensors which is not ideal
// torch::Tensor gemm_nt_cuda(torch::Tensor a, torch::Tensor b);
// torch::Tensor gemm_nn_cuda(torch::Tensor a, torch::Tensor b);
// torch::Tensor softmax_cuda(torch::Tensor x);

void run_gemm_nt(const int TILE_SIZE, dim3 &blocks_per_grid, dim3 &threads_per_block, float *a_mat, float *b_mat, float *out_mat, int M, int N, int K);
void run_gemm_nn(const int TILE_SIZE, dim3 blocks_per_grid, dim3 threads_per_block, float *a_mat, float *b_mat, float *out_mat, int M, int N, int K);
void run_softmax(dim3 blocks_per_grid, dim3 threads_per_block, float *inp, float *outp, int NUM_ROW, int NUM_COL);

__global__ void scale_and_causal_mask_batched(float *mat, int rows, int cols, float scale, bool causal) {
    int b = blockIdx.x;
    int row = blockIdx.y;
    int tid = threadIdx.x;

    if (row >= rows) return;

    int stride = rows * cols;
    float *row_ptr = mat + b * stride + row * cols;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = row_ptr[i] * scale;
        if (causal && i > row) {
            val = -FLT_MAX;
        }
        row_ptr[i] = val;
    }
}

torch::Tensor naive_attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                   bool causal, double scale) {

  // Get qk (gemm_nt), apply scale and causal
  // then apply softmax
  // this is s
  // then get o_bh = s @ v (gemm_nn)
  // reshape o_bh and return

  const int TILE_SIZE = 16;

  int batch_size = q.size(0);
  int seq_len = q.size(1);
  int hidden_dim = q.size(2);
  int head_dim = q.size(-1);
  int num_heads = hidden_dim / head_dim;

  auto q_bh = q.view({batch_size, seq_len, num_heads, head_dim}).permute({0, 2, 1, 3}).contiguous().view({batch_size * num_heads, seq_len, head_dim});
  auto k_bh = k.view({batch_size, seq_len, num_heads, head_dim}).permute({0, 2, 1, 3}).contiguous().view({batch_size * num_heads, seq_len, head_dim});
  auto v_bh = v.view({batch_size, seq_len, num_heads, head_dim}).permute({0, 2, 1, 3}).contiguous().view({batch_size * num_heads, seq_len, head_dim});




  throw std::runtime_error("NotYetImplemented: naive_attention_cuda");
}
