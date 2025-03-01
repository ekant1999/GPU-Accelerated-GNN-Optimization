#pragma once
#include <cuda_runtime.h>

void spmm_csr_launch(
    const int* row_ptr,
    const int* col_idx,
    const float* values,
    const float* features,
    float* output,
    int num_rows,
    int feature_dim,
    cudaStream_t stream
);
