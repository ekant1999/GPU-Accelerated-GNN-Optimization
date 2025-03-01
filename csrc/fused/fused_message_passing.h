#pragma once
#include <torch/extension.h>

torch::Tensor fused_gcn_forward(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor features,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor degree_inv_sqrt
);
