"""Graph partitioning for multi-GPU (re-export)."""
from gnn_opt.distributed.graph_partition import partition_graph, compute_halo_nodes

__all__ = ['partition_graph', 'compute_halo_nodes']
