#include "spmm_kernel.h"
#include "spmm_internal.cuh"
#include "../utils/cuda_utils.h"

torch::Tensor custom_spmm_forward(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor values,
    torch::Tensor features
) {
    TORCH_CHECK(row_ptr.is_cuda() && col_idx.is_cuda() && values.is_cuda() && features.is_cuda());
    TORCH_CHECK(row_ptr.dtype() == torch::kInt32);
    TORCH_CHECK(features.dtype() == torch::kFloat32);
    int num_rows = row_ptr.size(0) - 1;
    int feature_dim = features.size(1);

    auto output = torch::zeros({num_rows, feature_dim}, features.options());

    const int* row_ptr_ptr = row_ptr.data_ptr<int>();
    const int* col_idx_ptr = col_idx.data_ptr<int>();
    const float* values_ptr = values.data_ptr<float>();
    const float* features_ptr = features.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    spmm_csr_launch(row_ptr_ptr, col_idx_ptr, values_ptr, features_ptr, output_ptr,
                    num_rows, feature_dim, stream);

    return output;
}
