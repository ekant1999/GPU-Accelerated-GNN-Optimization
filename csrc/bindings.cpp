#include <torch/extension.h>
#include "spmm/spmm_kernel.h"
#include "fused/fused_message_passing.h"
#include "reduction/scatter_reduce.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spmm_forward", &custom_spmm_forward, "Custom SpMM forward");
    m.def("fused_gcn_forward", &fused_gcn_forward, "Fused GCN forward");
    m.def("scatter_reduce_forward", &scatter_reduce_forward, "Scatter reduce forward");
}
