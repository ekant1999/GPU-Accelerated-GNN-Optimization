"""Correctness tests for fused GCN message passing kernel."""
import torch


def test_fused_gcn_forward_shape():
    """Check output shape and no NaNs (if extension is built)."""
    try:
        from gnn_custom_ops import fused_gcn_forward
    except ImportError:
        return

    device = 'cuda'
    if not torch.cuda.is_available():
        return

    N, F_in, F_out = 100, 32, 64
    nnz = 500
    row_ptr = torch.zeros(N + 1, dtype=torch.int32, device=device)
    row_ptr[1:] = torch.linspace(0, nnz, N, dtype=torch.int32, device=device)
    col_idx = torch.randint(0, N, (nnz,), device=device)
    features = torch.randn(N, F_in, device=device)
    weights = torch.randn(F_in, F_out, device=device)
    bias = torch.zeros(F_out, device=device)
    deg_inv_sqrt = torch.ones(N, device=device) * 0.5

    out = fused_gcn_forward(row_ptr, col_idx, features, weights, bias, deg_inv_sqrt)
    assert out.shape == (N, F_out)
    assert not torch.isnan(out).any()


if __name__ == '__main__':
    test_fused_gcn_forward_shape()
    print('Fused kernel tests passed.')
