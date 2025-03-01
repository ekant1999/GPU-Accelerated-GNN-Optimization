#include "spmm_internal.cuh"
#include "../utils/cuda_utils.h"

// CSR SpMM: one warp per row. Each thread in warp handles (feature_dim / 32) dimensions.
__global__ void spmm_csr_warp_per_row(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ features,
    float* __restrict__ output,
    int num_rows,
    int feature_dim
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id >= num_rows) return;

    int row_start = row_ptr[warp_id];
    int row_end = row_ptr[warp_id + 1];

    for (int f = lane_id; f < feature_dim; f += 32) {
        float sum = 0.0f;
        for (int e = row_start; e < row_end; e++) {
            int col = col_idx[e];
            float val = values[e];
            sum += val * features[col * feature_dim + f];
        }
        output[warp_id * feature_dim + f] = sum;
    }
}

void spmm_csr_launch(
    const int* row_ptr,
    const int* col_idx,
    const float* values,
    const float* features,
    float* output,
    int num_rows,
    int feature_dim,
    cudaStream_t stream
) {
    const int warps_per_block = 4;
    const int block_size = warps_per_block * 32;
    int num_warps = (num_rows + warps_per_block - 1) / warps_per_block;
    int num_blocks = (num_warps + warps_per_block - 1) / warps_per_block;

    spmm_csr_warp_per_row<<<num_blocks, block_size, 0, stream>>>(
        row_ptr, col_idx, values, features, output,
        num_rows, feature_dim
    );
}
