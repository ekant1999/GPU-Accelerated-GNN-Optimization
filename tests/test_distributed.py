"""Multi-GPU / distributed tests (smoke tests without MPI)."""
import os
import torch


def test_setup_distributed_single():
    """Without MPI, setup_distributed should return rank=0, world_size=1."""
    from gnn_opt.distributed import setup_distributed
    rank, world_size = setup_distributed()
    assert rank == 0
    assert world_size >= 1


def test_partition_graph():
    """Graph partitioning returns num_parts partitions."""
    from torch_geometric.data import Data
    from gnn_opt.distributed import partition_graph

    data = Data(
        x=torch.randn(100, 8),
        edge_index=torch.randint(0, 100, (2, 400)),
    )
    parts = partition_graph(data, 4)
    assert len(parts) == 4


def test_compute_halo_nodes():
    """Halo nodes contain neighbors outside local partition."""
    from gnn_opt.distributed import compute_halo_nodes

    local_node_ids = torch.tensor([0, 1, 2])
    # Edges: 0-1, 1-2, 2-3, 3-4 -> halo should include 3
    full_edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    halo = compute_halo_nodes(local_node_ids, full_edge_index)
    assert 3 in halo


if __name__ == '__main__':
    test_setup_distributed_single()
    test_partition_graph()
    test_compute_halo_nodes()
    print('Distributed tests passed.')
