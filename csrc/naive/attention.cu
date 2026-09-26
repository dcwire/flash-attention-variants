// Task 2: assemble the primitives. S = gemm_nt(Q, K) * scale ; causal mask ; P = softmax(S) ;
// O = gemm_nn(P, V). Three launches + a full (B, H, N, N) S in HBM - this is the memory-bound
// baseline the fused kernel is measured against. Fold the scale and the mask into the softmax
// kernel (or a tiny elementwise kernel) rather than a separate pass over S.
#include <cstdio>
#include <cfloat>
#include <ATen/ATen.h>  // not torch/extension.h: only bindings.cpp needs pybind
#include <cuda_runtime.h>
#include <stdexcept>

// This generates 3 extra tensors which is not ideal
// at::Tensor gemm_nt_cuda(at::Tensor a, at::Tensor b);
// at::Tensor gemm_nn_cuda(at::Tensor a, at::Tensor b);
// at::Tensor softmax_cuda(at::Tensor x);

void run_gemm_nt(const int TILE_SIZE, dim3 &blocks_per_grid, dim3 &threads_per_block, float *a_mat, float *b_mat, float *out_mat, int M, int N, int K);
void run_gemm_nn(const int TILE_SIZE, dim3 &blocks_per_grid, dim3 &threads_per_block, float *a_mat, float *b_mat, float *out_mat, int M, int N, int K);
void run_softmax(dim3 &blocks_per_grid, dim3 &threads_per_block, float *inp, float *outp, int NUM_ROW, int NUM_COL);

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

at::Tensor naive_attention_cuda(at::Tensor q, at::Tensor k, at::Tensor v,
                                   bool causal, double scale) {

  // Get qk (gemm_nt), apply scale and causal
  // then apply softmax
  // this is s
  // then get o_bh = s @ v (gemm_nn)
  // reshape o_bh and return

  auto options = q.options();
  const int TILE_SIZE = 16;
  // q -> BHND
  int batch_size = q.size(0);
  int seq_len = q.size(2);
  int hidden_dim = q.size(-1);
  int head_dim = q.size(-1);
  int num_heads = hidden_dim / head_dim;

  auto q_bh = q.contiguous().view({batch_size * num_heads, seq_len, head_dim});
  auto k_bh = k.contiguous().view({batch_size * num_heads, seq_len, head_dim});
  auto v_bh = v.contiguous().view({batch_size * num_heads, seq_len, head_dim});

  auto qk = at::empty({batch_size * num_heads, seq_len, seq_len}, options);
  auto s = at::empty_like(qk);
  auto o_bh = at::empty({batch_size * num_heads, seq_len, head_dim}, options);

  dim3 thread_per_block(TILE_SIZE, TILE_SIZE);
  dim3 blocks_per_grid(
      (seq_len + TILE_SIZE - 1) / TILE_SIZE,
      (seq_len + TILE_SIZE - 1) / TILE_SIZE,
      batch_size * num_heads);

  dim3 tb2(256);
  dim3 b2(batch_size * num_heads, seq_len);

  run_gemm_nt(TILE_SIZE, blocks_per_grid, thread_per_block, q_bh.data_ptr<float>(), k_bh.data_ptr<float>(), qk.data_ptr<float>(), seq_len, seq_len, head_dim);

  scale_and_causal_mask_batched<<<b2, tb2>>>(qk.data_ptr<float>(), seq_len, seq_len, scale, causal);

  run_softmax(b2, tb2, qk.data_ptr<float>(), s.data_ptr<float>(), seq_len, seq_len);
  run_gemm_nn(TILE_SIZE, blocks_per_grid, thread_per_block, s.data_ptr<float>(), v_bh.data_ptr<float>(), o_bh.data_ptr<float>(), seq_len, head_dim, seq_len);

  return o_bh.view({batch_size, num_heads, seq_len, head_dim}).contiguous();
}
