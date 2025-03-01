#!/usr/bin/env python3
"""Profiling script: run a few epochs for Nsight Systems/Compute.
Usage:
  nsys profile --stats=true python scripts/profile.py --epochs 5
  ncu --set full python scripts/profile.py --epochs 1
"""
import argparse
import torch
from gnn_opt.data.loader import get_dataset, prepare_csr_data
from gnn_opt.models import OptimizedGCN
from gnn_opt.utils.metrics import train_step_optimized, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ogbn-arxiv')
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()

    data, split_idx, num_classes = get_dataset(args.dataset)
    row_ptr, col_idx, values, degree_inv_sqrt = prepare_csr_data(data)
    data.edge_index_csr = (
        row_ptr.cuda(), col_idx.cuda(), values.cuda(), degree_inv_sqrt.cuda()
    )
    data.x = data.x.cuda()
    data.y = data.y.cuda()

    model = OptimizedGCN(
        data.x.size(1), 256, num_classes, num_layers=3,
    ).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(args.epochs):
        loss = train_step_optimized(model, data, optimizer, split_idx)
        if (epoch + 1) % 1 == 0:
            print(f'Epoch {epoch+1} loss={loss:.4f}')

    accs = evaluate(model, data, split_idx, use_csr=True)
    print(f'Test accuracy: {accs["test"]:.4f}')


if __name__ == '__main__':
    main()
