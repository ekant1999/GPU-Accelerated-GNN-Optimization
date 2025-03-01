#!/usr/bin/env python3
"""Train baseline GCN with stock PyG (no custom kernels)."""
import argparse
import torch
from gnn_opt.data.loader import get_dataset, prepare_csr_data
from gnn_opt.models import BaselineGCN
from gnn_opt.utils.metrics import train_step_baseline, evaluate


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    data, split_idx, num_classes = get_dataset(args.dataset, args.root)
    data = data.to(device)

    model = BaselineGCN(
        in_channels=data.x.size(1),
        hidden_channels=args.hidden,
        out_channels=num_classes,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        loss = train_step_baseline(model, data, optimizer, split_idx)
        if (epoch + 1) % 50 == 0:
            accs = evaluate(model, data, split_idx, use_csr=False)
            print(f'Epoch {epoch+1}: loss={loss:.4f} train={accs["train"]:.4f} valid={accs["valid"]:.4f} test={accs["test"]:.4f}')

    accs = evaluate(model, data, split_idx, use_csr=False)
    print(f'Final test accuracy: {accs["test"]:.4f}')


if __name__ == '__main__':
    main()
