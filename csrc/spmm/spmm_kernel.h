#pragma once
#include <torch/extension.h>

torch::Tensor custom_spmm_forward(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor values,
    torch::Tensor features
);
