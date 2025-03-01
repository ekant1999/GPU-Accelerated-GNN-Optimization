#!/usr/bin/env python3
"""Train optimized GCN with custom CUDA kernels."""
import argparse
import torch
from gnn_opt.data.loader import get_dataset, prepare_csr_data
from gnn_opt.models import OptimizedGCN
from gnn_opt.utils.metrics import train_step_optimized, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ogbn-arxiv')
    parser.add_argument('--root', default='./data')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    data, split_idx, num_classes = get_dataset(args.dataset, args.root)
    row_ptr, col_idx, values, degree_inv_sqrt = prepare_csr_data(data)
    data.edge_index_csr = (
        row_ptr.cuda(), col_idx.cuda(), values.cuda(), degree_inv_sqrt.cuda()
    )
    data.x = data.x.cuda()
    data.y = data.y.cuda()

    model = OptimizedGCN(
        in_channels=data.x.size(1),
        hidden_channels=args.hidden,
        out_channels=num_classes,
        num_layers=args.layers,
        dropout=args.dropout,
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        loss = train_step_optimized(model, data, optimizer, split_idx)
        if (epoch + 1) % 50 == 0:
            accs = evaluate(model, data, split_idx, use_csr=True)
            print(f'Epoch {epoch+1}: loss={loss:.4f} train={accs["train"]:.4f} valid={accs["valid"]:.4f} test={accs["test"]:.4f}')

    accs = evaluate(model, data, split_idx, use_csr=True)
    print(f'Final test accuracy: {accs["test"]:.4f}')


if __name__ == '__main__':
    main()
