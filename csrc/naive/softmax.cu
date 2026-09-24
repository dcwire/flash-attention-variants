// Task 2: row-wise softmax over the last dimension.
// Suggested design: one thread block per row, each thread strides over the row; two block
// reductions (max, then sum of exp(x - max)); write exp(x - max) / sum. Accumulate in fp32 even
// for fp16 input. Rows can be long (N up to 16k) so don't assume a row fits in one block's threads.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>


__global__ void SOFTMAX_kernel_batched(float *inp, float *outp, int NUM_ROW, int NUM_COL) {
    extern __shared__ float buffer[];

    int tid = threadIdx.x;
    // Skip blockIdx.x (batchIndex) number of rows
    int row = NUM_ROW * blockIdx.x + blockIdx.y;
    int base_addr = row * NUM_COL;

    float local_max = -FLT_MAX;
    for (int i = tid; i < NUM_COL; i += blockDim.x) {
        local_max = fmaxf(local_max, inp[base_addr + i]);
    }
    buffer[tid] = local_max;

    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            buffer[tid] = fmaxf(buffer[tid], buffer[tid + stride]);
        }
    }

    __syncthreads();
    local_max = buffer[0];
    float local_sum = 0.0f;
    for (int i = tid; i < NUM_COL; i += blockDim.x) {
        local_sum += expf(inp[base_addr + i] - local_max);
    }

    __syncthreads();
    buffer[tid] = local_sum;
    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) buffer[tid] += buffer[tid + stride];
    }

    __syncthreads();
    local_sum = buffer[0];
    for (int i = tid; i < NUM_COL; i += blockDim.x) {
        float x = expf(inp[base_addr + i] - local_max);
        outp[base_addr + i] = x / local_sum;
    }


}

void run_softmax(dim3 &blocks_per_grid, dim3 &threads_per_block, float *inp, float *outp, int NUM_ROW, int NUM_COL) {
    SOFTMAX_kernel_batched<<<blocks_per_grid, threads_per_block, threads_per_block.x * sizeof(float)>>>(
        inp, outp, NUM_ROW, NUM_COL);
}
torch::Tensor softmax_cuda(torch::Tensor x) {

    // Idea is to do a reduce across a block which traverses a row
    // find the max in the row, then find total_sum(e^(row_value - max))
    // then write e^(row_value) / total_sum into the output

    // Assuming dim = -1 always
    x = x.contiguous();
    auto x_sizes = x.sizes();


    int N_BATCH;
    int N_ROW;
    int N_COL;

    if (x_sizes.size() == 2) {
        N_BATCH = 1;
        N_ROW = x_sizes[0];
        N_COL = x_sizes[1];
    } else {
        N_BATCH = x_sizes[0];
        N_ROW = x_sizes[1];
        N_COL = x_sizes[2];
    }

    dim3 threads_per_block(256);
    dim3 blocks_per_grid(N_BATCH, N_ROW);
    auto outp = torch::empty(x_sizes, x.options());

    SOFTMAX_kernel_batched<<<blocks_per_grid, threads_per_block, threads_per_block.x * sizeof(float)>>>(
        x.data_ptr<float>(), outp.data_ptr<float>(), N_ROW, N_COL);

    return outp;
}
