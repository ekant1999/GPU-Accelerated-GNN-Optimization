#include "scatter_reduce.h"
#include "../utils/cuda_utils.h"

__global__ void scatter_reduce_smem_kernel(
    const float* __restrict__ messages,
    float* __restrict__ output,
    const int* __restrict__ node_offsets,
    int num_nodes,
    int feature_dim
) {
    extern __shared__ float sdata[];

    int node = blockIdx.x;
    if (node >= num_nodes) return;

    int start = node_offsets[node];
    int end = node_offsets[node + 1];

    for (int f = 0; f < feature_dim; f++) {
        float val = 0.0f;
        for (int e = start + threadIdx.x; e < end; e += blockDim.x) {
            val += messages[e * feature_dim + f];
        }
        sdata[threadIdx.x] = val;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                sdata[threadIdx.x] += sdata[threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            output[node * feature_dim + f] = sdata[0];
        }
        __syncthreads();
    }
}

torch::Tensor scatter_reduce_forward(
    torch::Tensor messages,
    torch::Tensor node_offsets,
    int num_nodes,
    int feature_dim
) {
    auto output = torch::zeros({num_nodes, feature_dim}, messages.options());
    int block_size = 256;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    scatter_reduce_smem_kernel<<<num_nodes, block_size, block_size * sizeof(float), stream>>>(
        messages.data_ptr<float>(),
        output.data_ptr<float>(),
        node_offsets.data_ptr<int>(),
        num_nodes,
        feature_dim
    );

    return output;
}
