"""Correctness tests for custom SpMM."""
import torch


def test_spmm_correctness():
    """Verify custom SpMM matches PyTorch sparse matmul (if extension is built)."""
    try:
        from gnn_custom_ops import spmm_forward
    except ImportError:
        return  # skip if extension not built

    N, F = 500, 64
    density = 0.01
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        return

    # Random sparse matrix in CSR
    adj = (torch.rand(N, N, device=device) < density).float()
    adj = adj + torch.eye(N, device=device)  # avoid empty rows
    adj_csr = adj.to_sparse_csr()
    features = torch.randn(N, F, device=device, dtype=torch.float32)

    # Reference: PyTorch sparse matmul (COO)
    adj_coo = adj_csr.to_sparse()
    expected = torch.sparse.mm(adj_coo, features)

    # Custom kernel
    actual = spmm_forward(
        adj_csr.crow_indices().int(),
        adj_csr.col_indices().int(),
        adj_csr.values(),
        features
    )

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def test_spmm_empty_rows():
    """Nodes with no neighbors (after removing self-loops) - row of zeros."""
    try:
        from gnn_custom_ops import spmm_forward
    except ImportError:
        return
    device = 'cuda'
    if not torch.cuda.is_available():
        return
    # 3 nodes, only edge 0->1; node 2 has no in-edges. CSR row_ptr [0,1,2,2]
    row_ptr = torch.tensor([0, 1, 2, 2], device=device, dtype=torch.int32)
    col_idx = torch.tensor([1, 0], device=device, dtype=torch.int32)
    values = torch.ones(2, device=device)
    features = torch.ones(3, 4, device=device)
    out = spmm_forward(row_ptr, col_idx, values, features)
    assert out.shape == (3, 4)
    # Row 2 should be zero
    assert (out[2] == 0).all()


if __name__ == '__main__':
    test_spmm_correctness()
    test_spmm_empty_rows()
    print('SpMM tests passed.')
