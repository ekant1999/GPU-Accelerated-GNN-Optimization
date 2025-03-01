#!/usr/bin/env python3
"""Multi-GPU training with MPI and gradient AllReduce."""
import argparse
import torch
import torch.nn.functional as F
from gnn_opt.data.loader import get_dataset, prepare_csr_data
from gnn_opt.models import OptimizedGCN
from gnn_opt.utils.metrics import evaluate
from gnn_opt.distributed import setup_distributed
from gnn_opt.distributed.data_parallel import allreduce_gradients


def train_step(model, data, optimizer, split_idx, rank, world_size):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index_csr)
    loss = F.cross_entropy(out[split_idx['train']], data.y[split_idx['train']].squeeze())
    loss.backward()
    allreduce_gradients(model)
    optimizer.step()
    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ogbn-arxiv')
    parser.add_argument('--root', default='./data')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layers', type=int, default=3)
    args = parser.parse_args()

    rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    data, split_idx, num_classes = get_dataset(args.dataset, args.root)
    row_ptr, col_idx, values, degree_inv_sqrt = prepare_csr_data(data)
    data.edge_index_csr = (
        row_ptr.to(device), col_idx.to(device),
        values.to(device), degree_inv_sqrt.to(device)
    )
    data.x = data.x.to(device)
    data.y = data.y.to(device)

    model = OptimizedGCN(
        in_channels=data.x.size(1),
        hidden_channels=args.hidden,
        out_channels=num_classes,
        num_layers=args.layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    for epoch in range(args.epochs):
        loss = train_step(model, data, optimizer, split_idx, rank, world_size)
        if rank == 0 and (epoch + 1) % 50 == 0:
            accs = evaluate(model, data, split_idx, use_csr=True)
            print(f'Epoch {epoch+1}: loss={loss:.4f} test={accs["test"]:.4f}')

    if rank == 0:
        accs = evaluate(model, data, split_idx, use_csr=True)
        print(f'Final test accuracy: {accs["test"]:.4f}')

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
