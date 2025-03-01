#include "fused_message_passing.h"
#include "../utils/cuda_utils.h"

__global__ void fused_gcn_message_passing_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ node_features,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int num_nodes,
    int in_features,
    int out_features,
    const float* __restrict__ degree_inv_sqrt
) {
    int node = blockIdx.x;
    if (node >= num_nodes) return;

    int row_start = row_ptr[node];
    int row_end = row_ptr[node + 1];
    float norm_self = degree_inv_sqrt[node];

    for (int out_f = threadIdx.x; out_f < out_features; out_f += blockDim.x) {
        float agg = 0.0f;

        for (int e = row_start; e < row_end; e++) {
            int neighbor = col_idx[e];
            float norm_neighbor = degree_inv_sqrt[neighbor];
            float edge_weight = norm_self * norm_neighbor;

            float msg = 0.0f;
            for (int in_f = 0; in_f < in_features; in_f++) {
                msg += node_features[neighbor * in_features + in_f]
                     * weights[in_f * out_features + out_f];
            }
            agg += edge_weight * msg;
        }

        float self_msg = 0.0f;
        for (int in_f = 0; in_f < in_features; in_f++) {
            self_msg += node_features[node * in_features + in_f]
                      * weights[in_f * out_features + out_f];
        }
        agg += norm_self * norm_self * self_msg;

        agg += bias[out_f];
        output[node * out_features + out_f] = fmaxf(agg, 0.0f);
    }
}

torch::Tensor fused_gcn_forward(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor features,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor degree_inv_sqrt
) {
    TORCH_CHECK(row_ptr.is_cuda() && col_idx.is_cuda() && features.is_cuda());
    TORCH_CHECK(weights.is_cuda() && bias.is_cuda() && degree_inv_sqrt.is_cuda());
    int num_nodes = row_ptr.size(0) - 1;
    int in_features = features.size(1);
    int out_features = weights.size(1);

    auto output = torch::empty({num_nodes, out_features}, features.options());

    int block_size = 256;
    if (out_features < block_size) block_size = (out_features + 31) / 32 * 32;
    if (block_size > 1024) block_size = 1024;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_gcn_message_passing_kernel<<<num_nodes, block_size, 0, stream>>>(
        row_ptr.data_ptr<int>(),
        col_idx.data_ptr<int>(),
        features.data_ptr<float>(),
        weights.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        num_nodes, in_features, out_features,
        degree_inv_sqrt.data_ptr<float>()
    );

    return output;
}
