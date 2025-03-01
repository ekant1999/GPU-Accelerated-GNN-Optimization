#pragma once
#include <torch/extension.h>

torch::Tensor scatter_reduce_forward(
    torch::Tensor messages,
    torch::Tensor node_offsets,
    int num_nodes,
    int feature_dim
);
