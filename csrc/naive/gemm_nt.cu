// Task 2: C[b, m, n] = sum_k A[b, m, k] * B[b, n, k]      (A @ B^T, batched over leading dims)
// Used for S = Q K^T. Start with a 16x16 shared-memory tiled GEMM with bounds checks (N is not
// a multiple of the tile in tests). The stretch goal swaps the inner product for mma.sync
// (m16n8k16 fp16 fragments) - keep the tiling so only the inner loop changes.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>

template <int TILE_SIZE>
__global__ GEMM_NT_kernel_batched(float *a_mat, float *b_mat, float *out_mat, int M, int N, int K) {
    __shared__ float a_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float b_tile[TILE_SIZE][TILE_SIZE];

    int b_idx = blockIdx.z;
    int a_base_addr = b_idx * M * K;
    int b_base_addr = b_idx * N * K;
    int o_base_addr = b_idx * M * N;

    int t_y = blockIdx.y * TILE_SIZE;
    int t_x = blockIdx.x * TILE_SIZE;

    int r = threadIdx.y;
    int c = threadIdx.x;

    float ans = 0.0f;
    for (int k=0; k<((K + TILE_SIZE - 1) / TILE_SIZE); k++) {
        int t_k = k * TILE_SIZE;

        __syncthreads();
        if ((t_y + r) < M && (t_k + c) < K) a_tile[r][c] = a_mat[a_base_addr + (t_y + r) * K + (t_k + c)];
        else a_tile[r][c] = 0.0f;

        if ((t_x + r) < N && (t_k + c) < K) b_tile[r][c] = b_mat[b_base_addr + (t_x + r) * N + (t_k + c)];
        else b_tile[r][c] = 0.0f;
        __syncthreads();

        for (int i=0; i<TILE_SIZE; i++) {
            ans += a_tile[r][i] * b_tile[c][i];
        }

    }

    if ((t_y + r) < M && (t_x + c) < N) {
        out_mat[o_base_addr + (t_y + r) * N + (t_x + c)] = ans;
    }

}


void run_gemm_nt(const int TILE_SIZE, dim3 &blocks_per_grid, dim3 &threads_per_block, float *a_mat, float *b_mat, float *out_mat, int M, int N, int K) {
    GEMM_NT_kernel_batched<TILE_SIZE><<<blocks_per_grid, threads_per_block>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), M, N, K);
}

torch::Tensor gemm_nt_cuda(torch::Tensor a, torch::Tensor b) {

    a = a.contiguous();
    b = b.contiguous();

    auto a_size = a.sizes();
    auto b_size = b.sizes();
    int N_BATCH = a_size[0];
    int M = a_size[1], N = b_size[1], K = a_size[2];

    auto out = torch::empty({N_BATCH, M, N}, a.options());
    const int TILE_SIZE = 16;
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dim3 blocks_per_grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE,
        N_BATCH);
    GEMM_NT_kernel_batched<TILE_SIZE><<<blocks_per_grid, threads_per_block>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), M, N, K);
    return out;
}
