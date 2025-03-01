#!/usr/bin/env python3
"""Full benchmarking suite: baseline vs optimized, timing and accuracy."""
import argparse
import time
import numpy as np
import torch
from gnn_opt.data.loader import get_dataset, prepare_csr_data
from gnn_opt.models import BaselineGCN, OptimizedGCN
from gnn_opt.utils.metrics import train_step_baseline, train_step_optimized, evaluate


def benchmark_baseline(data, split_idx, num_classes, num_seeds=3, num_epochs=100):
    results = {'test_accs': [], 'epoch_times': [], 'peak_memory': []}
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        model = BaselineGCN(
            data.x.size(1), 256, num_classes, num_layers=3, dropout=0.5
        ).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        torch.cuda.reset_peak_memory_stats()
        epoch_times = []
        for epoch in range(num_epochs):
            t0 = time.perf_counter()
            train_step_baseline(model, data, optimizer, split_idx)
            torch.cuda.synchronize()
            epoch_times.append(time.perf_counter() - t0)
        accs = evaluate(model, data, split_idx, use_csr=False)
        results['test_accs'].append(accs['test'])
        results['epoch_times'].append(np.mean(epoch_times))
        results['peak_memory'].append(torch.cuda.max_memory_allocated() / 1e9)
    return results


def benchmark_optimized(data, split_idx, num_classes, num_seeds=3, num_epochs=100):
    results = {'test_accs': [], 'epoch_times': [], 'peak_memory': []}
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        model = OptimizedGCN(
            data.x.size(1), 256, num_classes, num_layers=3, dropout=0.5
        ).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        torch.cuda.reset_peak_memory_stats()
        epoch_times = []
        for epoch in range(num_epochs):
            t0 = time.perf_counter()
            train_step_optimized(model, data, optimizer, split_idx)
            torch.cuda.synchronize()
            epoch_times.append(time.perf_counter() - t0)
        accs = evaluate(model, data, split_idx, use_csr=True)
        results['test_accs'].append(accs['test'])
        results['epoch_times'].append(np.mean(epoch_times))
        results['peak_memory'].append(torch.cuda.max_memory_allocated() / 1e9)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ogbn-arxiv')
    parser.add_argument('--root', default='./data')
    parser.add_argument('--num-seeds', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    print('Loading data...')
    data, split_idx, num_classes = get_dataset(args.dataset, args.root)
    data = data.cuda()

    print('Benchmarking baseline (PyG)...')
    baseline = benchmark_baseline(data, split_idx, num_classes, args.num_seeds, args.epochs)

    row_ptr, col_idx, values, degree_inv_sqrt = prepare_csr_data(data)
    data.edge_index_csr = (row_ptr.cuda(), col_idx.cuda(), values.cuda(), degree_inv_sqrt.cuda())

    print('Benchmarking optimized (custom CUDA)...')
    try:
        optimized = benchmark_optimized(data, split_idx, num_classes, args.num_seeds, args.epochs)
    except Exception as e:
        print(f'Optimized run failed (extensions may not be built): {e}')
        optimized = None

    print('\n--- Results ---')
    print(f'Baseline  test_acc: {np.mean(baseline["test_accs"]):.4f} ± {np.std(baseline["test_accs"]):.4f}')
    print(f'Baseline  epoch_time: {np.mean(baseline["epoch_times"]):.4f}s')
    print(f'Baseline  peak_memory: {np.mean(baseline["peak_memory"]):.2f} GB')
    if optimized is not None:
        print(f'Optimized test_acc: {np.mean(optimized["test_accs"]):.4f} ± {np.std(optimized["test_accs"]):.4f}')
        print(f'Optimized epoch_time: {np.mean(optimized["epoch_times"]):.4f}s')
        print(f'Optimized peak_memory: {np.mean(optimized["peak_memory"]):.2f} GB')
        speedup = np.mean(baseline['epoch_times']) / np.mean(optimized['epoch_times'])
        print(f'Speedup (epoch time): {speedup:.2f}x')


if __name__ == '__main__':
    main()
