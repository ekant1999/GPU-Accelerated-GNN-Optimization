"""Data loading and preprocessing for node classification."""
import numpy as np
import torch
import scipy.sparse as sp
from torch_geometric.utils import to_scipy_sparse_matrix


def get_dataset(name='ogbn-arxiv', root='./data'):
    """Load a PyG node classification dataset with ~2.3M edges."""
    if name == 'ogbn-arxiv':
        # PyTorch 2.6+ defaults to weights_only=True; OGB processed data uses PyG types
        # Use weights_only=False for this trusted dataset load only
        from ogb.nodeproppred import PygNodePropPredDataset
        _orig_load = torch.load
        try:
            def _load_trusted(*args, **kwargs):
                kwargs['weights_only'] = False
                return _orig_load(*args, **kwargs)
            torch.load = _load_trusted
            dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=root)
        finally:
            torch.load = _orig_load
        data = dataset[0]
        # OGB uses split_idx dict
        split_idx = dataset.get_idx_split()
        data.split_idx = split_idx
        return data, split_idx, dataset.num_classes
    raise ValueError(f"Unknown dataset: {name}")


def prepare_csr_data(data, add_self_loops=True):
    """Convert PyG data to CSR format with GCN normalization.
    Returns (row_ptr, col_idx, values, degree_inv_sqrt) for use with custom kernels.
    """
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
    if add_self_loops:
        adj = adj + sp.eye(adj.shape[0])
    adj_csr = adj.tocsr()

    row_ptr = torch.from_numpy(adj_csr.indptr).to(torch.int32)
    col_idx = torch.from_numpy(adj_csr.indices).to(torch.int32)

    # GCN normalization: D^{-1/2} A D^{-1/2}
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = torch.from_numpy(np.power(deg, -0.5)).float()
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0

    # Row index for each non-zero
    nnz = col_idx.size(0)
    rows = torch.repeat_interleave(
        torch.arange(adj_csr.shape[0], dtype=torch.long),
        torch.diff(torch.from_numpy(adj_csr.indptr))
    )
    row_degs = deg_inv_sqrt[rows]
    col_degs = deg_inv_sqrt[col_idx]
    values = (row_degs * col_degs).float()

    return row_ptr, col_idx, values, deg_inv_sqrt
