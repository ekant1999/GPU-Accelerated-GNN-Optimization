"""Custom GNN conv layer using CUDA kernels."""
import torch
from torch.autograd import Function

try:
    from gnn_custom_ops import spmm_forward, fused_gcn_forward
except ImportError:
    spmm_forward = None
    fused_gcn_forward = None


def _transpose_csr(row_ptr, col_idx, values, num_rows, num_cols):
    """Transpose a CSR matrix: (row_ptr, col_idx, values) -> (row_ptr_t, col_idx_t, values_t)."""
    device = row_ptr.device
    nnz = col_idx.size(0)
    # Row index for each non-zero: row i has entries from row_ptr[i] to row_ptr[i+1]
    rows = torch.repeat_interleave(
        torch.arange(num_rows, device=device, dtype=torch.long),
        row_ptr[1:] - row_ptr[:-1]
    )
    # COO of A: (rows, col_idx), values. COO of A^T: (col_idx, rows), values.
    # Sort by col_idx to get rows of A^T in order.
    perm = torch.argsort(col_idx)
    sorted_col = col_idx[perm]
    sorted_row = rows[perm]
    sorted_val = values[perm]
    # Build row_ptr for A^T: row i has entries where sorted_col == i
    row_counts = torch.bincount(sorted_col, minlength=num_cols)
    row_ptr_t = torch.cat([torch.tensor([0], device=device, dtype=row_ptr.dtype), row_counts.cumsum(0)])
    col_idx_t = sorted_row.to(torch.int32)
    values_t = sorted_val
    return row_ptr_t, col_idx_t, values_t


class CustomSpMMFunction(Function):
    @staticmethod
    def forward(ctx, row_ptr, col_idx, values, features, num_nodes):
        if spmm_forward is None:
            raise RuntimeError("gnn_custom_ops not built. Run: pip install -e .")
        ctx.save_for_backward(row_ptr, col_idx, values, features)
        ctx.num_nodes = num_nodes
        return spmm_forward(row_ptr, col_idx, values, features)

    @staticmethod
    def backward(ctx, grad_output):
        row_ptr, col_idx, values, features = ctx.saved_tensors
        n = ctx.num_nodes
        t_row_ptr, t_col_idx, t_values = _transpose_csr(row_ptr, col_idx, values, n, n)
        grad_features = spmm_forward(t_row_ptr, t_col_idx, t_values, grad_output)
        return None, None, None, grad_features, None


class CustomGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(in_channels, out_channels))
        self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, x, edge_index_csr):
        row_ptr, col_idx, values, degree_inv_sqrt = edge_index_csr
        if fused_gcn_forward is None:
            raise RuntimeError("gnn_custom_ops not built. Run: pip install -e .")
        return fused_gcn_forward(
            row_ptr, col_idx, x, self.weight, self.bias, degree_inv_sqrt
        )
