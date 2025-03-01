"""Multi-GPU data parallel training with gradient AllReduce."""
import torch
import torch.nn as nn
import torch.distributed as dist


def allreduce_gradients(model):
    """AllReduce gradients across all processes."""
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size
