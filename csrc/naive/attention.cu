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

  std::printf("Printing q shape: \n");
  for (auto &s: q.sizes()) {
      std::printf("%d ", s);
  }

  std::printf("\n Printing k shape: \n");
  for (auto &s: k.sizes()) {
      std::printf("%d ", s);
  }

  std::printf("\n Printing v shape: \n");
  for (auto &s: v.sizes()) {
      std::printf("%d ", s);
  }

  throw std::runtime_error("NotYetImplemented: naive_attention_cuda");
}
