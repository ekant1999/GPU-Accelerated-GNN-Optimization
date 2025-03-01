"""Graph partitioning for multi-GPU training."""
import torch


def partition_graph(data, num_parts):
    """Partition graph across GPUs (simplified: sequential split by node ID).
    For METIS-based partitioning, use torch_geometric.loader.ClusterData.
    """
    try:
        from torch_geometric.loader import ClusterData
        cluster_data = ClusterData(data, num_parts=num_parts)
        return [cluster_data[i] for i in range(num_parts)]
    except Exception:
        # Fallback: sequential split
        n = data.num_nodes
        part_size = (n + num_parts - 1) // num_parts
        partitions = []
        for i in range(num_parts):
            start = i * part_size
            end = min((i + 1) * part_size, n)
            node_ids = torch.arange(start, end, device=data.x.device)
            partitions.append(node_ids)
        return partitions


def compute_halo_nodes(local_node_ids, full_edge_index):
    """Find boundary nodes that need feature exchange between partitions."""
    local_nodes = set(local_node_ids.tolist())
    halo_nodes = set()
    src, dst = full_edge_index[0], full_edge_index[1]
    for s, d in zip(src.tolist(), dst.tolist()):
        if s in local_nodes and d not in local_nodes:
            halo_nodes.add(d)
        if d in local_nodes and s not in local_nodes:
            halo_nodes.add(s)
    return halo_nodes
