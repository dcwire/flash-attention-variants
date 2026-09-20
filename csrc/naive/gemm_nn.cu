// Task 2: C[b, m, n] = sum_k A[b, m, k] * B[b, k, n]      (A @ B, batched over leading dims)
// Used for O = P V. Same tiling as gemm_nt; the difference is only how B tiles are loaded into
// shared memory (row-major K x N slice instead of N x K), which is exactly the fragment layout
// question you will hit again for the S.V product inside the fused kernel.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>

template <int TILE_SIZE>
__global__ void GEMM_NN_kernel_batched(float *a_mat, float *b_mat, float *out_mat, int M, int N, int K) {

    __shared__ float a_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float b_tile[TILE_SIZE][TILE_SIZE];

    // Batch index
    const int b_idx = blockIdx.z;
    // Skip elements of a batch depending on index
    const int o_base_addr = b_idx * M * N;
    const int a_base_addr = b_idx * M * K;
    const int b_base_addr = b_idx * N * K;

    // Row and column of the tile
    int t_y = blockIdx.y * TILE_SIZE;
    int t_x = blockIdx.x * TILE_SIZE;

    // Row and column inside the tile
    int r = threadIdx.y;
    int c = threadIdx.x;

    float ans = 0.0f;

    for (int k=0; k < ((K + TILE_SIZE - 1) / TILE_SIZE); k++) {

        int t_k = k * TILE_SIZE;

        __syncthreads();
        if ((t_y + r) < M && (t_k + c) < K)
            a_tile[r][c] = a_mat[a_base_addr + (t_y + r) * K + (t_k + c)];
        else
            a_tile[r][c] = 0.0f;

        if ((t_k + r) < K && (t_x + c) < N)
            b_tile[r][c] = b_mat[b_base_addr + (t_k + r) * N + (t_x + c)];
        else
            b_tile[r][c] = 0.0f;
        __syncthreads();

        for (int i=0; i<TILE_SIZE; i++) {
            ans += a_tile[r][i] * b_tile[i][c];
        }
    }

    if ((t_y + r) < M && (t_x + c) < N)
        out_mat[o_base_addr + (t_y + r) * N + (t_x + c)] = ans;


}
torch::Tensor gemm_nn_cuda(torch::Tensor a, torch::Tensor b) {
    a = a.contiguous();
    b = b.contiguous();

    auto a_size = a.sizes();
    auto b_size = b.sizes();
    int N_BATCH = a_size[0];
    int M = a_size[1], N = b_size[2], K = a_size[2];

    auto out = torch::empty({N_BATCH, M, N}, a.options());
    const int TILE_SIZE = 16;
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dim3 blocks_per_grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE,
        N_BATCH
    );

    GEMM_NN_kernel_batched<TILE_SIZE>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), M, N, K);

    return out;

  throw std::runtime_error("NotYetImplemented: gemm_nn_cuda");
}
