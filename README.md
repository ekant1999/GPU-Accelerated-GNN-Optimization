# GPU-Accelerated GNN Optimization

Project goals (from plan):

1. **~6.8% improvement** in node classification accuracy on a ~2.3M-edge benchmark (ogbn-arxiv) via custom CUDA kernels for sparse operations with PyTorch Geometric.
2. **~2.4x faster** training on multi-GPU (e.g. 4 GPUs) using distributed data parallelism and an optimized backend.

## Tech Stack

- **C++17** + **CUDA 11.8+** for custom SpMM and fused message-passing kernels  
- **PyTorch 2.0+** and **PyTorch Geometric (PyG)** for models and data  
- **MPI (mpi4py)** for multi-GPU launch and communication  
- **CMake / PyTorch cpp_extension** for building CUDA extensions

## Setup

```bash
# 1. Create env and install Python deps
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Build CUDA extensions (requires CUDA toolkit and GPU)
pip install -e .
# Or: python setup.py build_ext --inplace
```

## Usage

```bash
# Baseline (stock PyG)
python scripts/train_baseline.py --dataset ogbn-arxiv --epochs 500

# Optimized (custom CUDA kernels)
python scripts/train_optimized.py --dataset ogbn-arxiv --epochs 500

# Multi-GPU (e.g. 4 GPUs)
mpirun -np 4 python scripts/train_distributed.py --dataset ogbn-arxiv --epochs 500

# Benchmark
python scripts/benchmark.py --num-seeds 3 --epochs 100

# Profiling
nsys profile --stats=true python scripts/profile.py --epochs 5
ncu --set full python scripts/profile.py --epochs 1
```

## Project Layout

- `csrc/` — C++/CUDA: SpMM kernel, fused GCN kernel, scatter reduce, bindings  
- `gnn_opt/` — Python package: models (GCN baseline + optimized), layers, data loaders, distributed helpers  
- `scripts/` — Training and benchmarking scripts  
- `configs/default.yaml` — Default hyperparameters  
- `tests/` — Correctness tests for SpMM and fused kernel

## Dataset

Default: **ogbn-arxiv** (~169K nodes, ~2.3M directed edges, 128-dim features, 40 classes). Data is downloaded to `./data` on first run.

**Note:** If you see a `Weights only load failed` or `Unsupported global` error when loading OGB data with PyTorch 2.6+, either use PyTorch < 2.6 or allowlist PyG types before loading, e.g. `torch.serialization.add_safe_globals([...])` as suggested in the error message.

.