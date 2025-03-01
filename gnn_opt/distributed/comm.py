"""MPI and PyTorch distributed helpers."""
import os
import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize PyTorch distributed with NCCL using MPI rank/size."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()
    except ImportError:
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))

    if world_size > 1:
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, world_size
