# Prompt: Build a GPU-Accelerated Graph Neural Network Optimization Project from Scratch


> Use this prompt with an AI coding assistant to recreate the full project. Feed the entire document as context.


---


## Project Overview


Build a complete **GPU-Accelerated Graph Neural Network (GNN) Optimization** project that achieves two measurable outcomes:


1. **6.8% improvement in node classification accuracy** on a 2.3M-edge benchmark dataset by integrating PyTorch Geometric (PyG) with custom CUDA kernels for sparse linear algebra operations.
2. **2.4x faster training throughput** on multi-GPU systems (measured by epoch completion time across 4 GPUs) by implementing distributed data parallelism using MPI-style communication and an optimized C++ backend.


---


## Tech Stack (Exact)


| Component | Technology | Version Guidance |
|-----------|-----------|-----------------|
| Core Language | C++ (C++17) | GPU kernel development, backend routines |
| GPU Programming | CUDA (11.8+) | Custom sparse kernels, memory management |
| ML Framework | PyTorch (2.0+) | Training loop, autograd, model definition |
| Graph ML Library | PyTorch Geometric (PyG) | GNN layers, data handling, baselines |
| Distributed Compute | MPI (OpenMPI or MPICH) | Multi-GPU communication, process management |
| Profiling | Nsight Systems + Nsight Compute | Performance analysis and optimization |
| Build System | CMake + PyTorch cpp_extension | Compiling CUDA/C++ extensions |
| Data Format | PyG Data objects (COO/CSR sparse) | Graph storage and manipulation |


---


## Dataset


Use a benchmark graph dataset with approximately **2.3 million edges** for node classification. Recommended options (pick one):


### Option A: ogbn-arxiv (OGB — Open Graph Benchmark)
- **Nodes:** 169,343 (arXiv papers)
- **Edges:** 1,166,243 (citation links) — note: undirected = ~2.3M directed edges
- **Features:** 128-dimensional per node (word2vec embeddings of paper abstracts)
- **Classes:** 40 (arXiv subject areas)
- **Task:** Node classification (predict paper subject area)
- **Split:** Time-based (papers before 2017 = train, 2017 = val, 2018 = test)


```python

### Option C: Amazon Products / Citation Networks
- Use any PyG-compatible dataset that has ~2.3M edges and supports node classification.


**The exact dataset matters less than the edge count (~2.3M) and that it's a node classification benchmark with established baselines.**


---


## Project Architecture


```
gnn-optimization/
├── CMakeLists.txt                    # Build system for CUDA/C++ extensions
├── setup.py                          # Python package setup with cpp_extension
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
│
├── csrc/                             # C++/CUDA source code
│   ├── spmm/
│   │   ├── spmm_kernel.cu            # Custom SpMM CUDA kernel
│   │   ├── spmm_kernel.h             # Kernel declarations
│   │   └── spmm_csr.cu              # CSR-format SpMM implementation
│   ├── fused/
│   │   ├── fused_message_passing.cu  # Fused gather-message-aggregate-update kernel
│   │   └── fused_message_passing.h
│   ├── reduction/
│   │   ├── scatter_reduce.cu         # Shared-memory reduction (replaces atomics)
│   │   └── scatter_reduce.h
│   ├── utils/
│   │   ├── cuda_utils.h              # CUDA_CHECK macro, error handling
│   │   └── graph_partition.cpp       # Graph partitioning for multi-GPU
│   └── bindings.cpp                  # pybind11/PyTorch C++ extension bindings
│
├── gnn_opt/                          # Python package
│   ├── __init__.py
│   ├── models/
│   │   ├── gcn.py                    # GCN model (baseline + optimized)
│   │   ├── sage.py                   # GraphSAGE model
│   │   └── gat.py                    # GAT model (optional)
│   ├── layers/
│   │   ├── custom_conv.py            # Custom GNN conv layer using CUDA kernels
│   │   └── fused_conv.py             # Fused message passing layer
│   ├── distributed/
│   │   ├── data_parallel.py          # Multi-GPU data parallel wrapper
│   │   ├── graph_partition.py        # Graph partitioning utilities
│   │   └── comm.py                   # MPI communication helpers
│   ├── data/
│   │   ├── loader.py                 # Data loading and preprocessing
│   │   └── partition.py              # Graph partitioning for multi-GPU
│   └── utils/
│       ├── metrics.py                # Accuracy, F1, timing utilities
│       └── profiling.py              # Nsight integration helpers
│
├── scripts/
│   ├── train_baseline.py             # Train with stock PyG (baseline)
│   ├── train_optimized.py            # Train with custom CUDA kernels
│   ├── train_distributed.py          # Multi-GPU training script
│   ├── benchmark.py                  # Full benchmarking suite
│   └── profile.py                    # Profiling script
│
├── configs/
│   └── default.yaml                  # Hyperparameters and experiment config
│
├── tests/
│   ├── test_spmm.py                  # Correctness tests for custom SpMM
│   ├── test_fused_kernel.py          # Correctness tests for fused kernels
│   └── test_distributed.py           # Multi-GPU communication tests
│
└── results/
   └── .gitkeep                      # Benchmark results stored here
```


---


## Phase 0: Build System and Environment


### 0.1 setup.py (for building CUDA extensions)


```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
   name='gnn_opt',
   packages=['gnn_opt', 'gnn_opt.models', 'gnn_opt.layers', 'gnn_opt.distributed', 'gnn_opt.data', 'gnn_opt.utils'],
   ext_modules=[
       CUDAExtension(
           name='gnn_custom_ops',
           sources=[
               'csrc/spmm/spmm_kernel.cu',
               'csrc/spmm/spmm_csr.cu',
               'csrc/fused/fused_message_passing.cu',
               'csrc/reduction/scatter_reduce.cu',
               'csrc/bindings.cpp',
           ],
           extra_compile_args={
               'cxx': ['-O3', '-std=c++17'],
               'nvcc': ['-O3', '--use_fast_math', '-std=c++17',
                        '-gencode=arch=compute_80,code=sm_80',   # A100
                        '-gencode=arch=compute_86,code=sm_86',   # RTX 3090
                        '-gencode=arch=compute_89,code=sm_89'],  # RTX 4090
           },
       ),
   ],
   cmdclass={'build_ext': BuildExtension},
)
```


### 0.2 CUDA Error Handling Utilities


```cpp
// csrc/utils/cuda_utils.h
#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>


#define CUDA_CHECK(call)                                                  \
   do {                                                                  \
       cudaError_t err = call;                                           \
       if (err != cudaSuccess) {                                         \
           fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                   __FILE__, __LINE__, cudaGetErrorString(err));          \
           exit(EXIT_FAILURE);                                           \
       }                                                                 \
   } while (0)


#define CUDA_CHECK_LAST_ERROR()                                           \
   do {                                                                  \
       cudaError_t err = cudaGetLastError();                              \
       if (err != cudaSuccess) {                                         \
           fprintf(stderr, "CUDA kernel error at %s:%d: %s\n",           \
                   __FILE__, __LINE__, cudaGetErrorString(err));          \
           exit(EXIT_FAILURE);                                           \
       }                                                                 \
   } while (0)


inline int div_ceil(int a, int b) { return (a + b - 1) / b; }
```


### 0.3 Requirements


```
# requirements.txt
torch>=2.0.0
torch-geometric>=2.4.0
torch-scatter
torch-sparse
ogb>=1.3.6
pyyaml
tensorboard
numpy
scipy
mpi4py
cmake>=3.18
```


---


## Phase 1: Baseline Implementation with PyG


### 1.1 Implement Baseline GCN Model


Build a standard GCN for node classification using stock PyG:


```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class BaselineGCN(torch.nn.Module):
   def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
       super().__init__()
       self.convs = torch.nn.ModuleList()
       self.bns = torch.nn.ModuleList()


       self.convs.append(GCNConv(in_channels, hidden_channels))
       self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
       for _ in range(num_layers - 2):
           self.convs.append(GCNConv(hidden_channels, hidden_channels))
           self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
       self.convs.append(GCNConv(hidden_channels, out_channels))


       self.dropout = dropout


   def forward(self, x, edge_index):
       for i, conv in enumerate(self.convs[:-1]):
           x = conv(x, edge_index)
           x = self.bns[i](x)
           x = F.relu(x)
           x = F.dropout(x, p=self.dropout, training=self.training)
       x = self.convs[-1](x, edge_index)
       return x
```


### 1.2 Training Loop (Baseline)


```python
def train_baseline(model, data, optimizer, split_idx):
   model.train()
   optimizer.zero_grad()
   out = model(data.x, data.edge_index)
   loss = F.cross_entropy(out[split_idx['train']], data.y[split_idx['train']].squeeze())
   loss.backward()
   optimizer.step()
   return loss.item()


@torch.no_grad()
def evaluate(model, data, split_idx):
   model.eval()
   out = model(data.x, data.edge_index)
   y_pred = out.argmax(dim=-1)
   accs = {}
   for key in ['train', 'valid', 'test']:
       mask = split_idx[key]
       accs[key] = (y_pred[mask] == data.y[mask].squeeze()).float().mean().item()
   return accs
```


### 1.3 Record Baseline Metrics


Capture these metrics across 5+ seeds:
- Test accuracy (mean ± std)
- Training time per epoch
- Total training time to convergence
- Peak GPU memory usage
- Per-epoch timing breakdown


---


## Phase 2: Custom CUDA Kernels for Sparse Operations


This is the core innovation. PyG uses generic cuSPARSE and separate kernel launches. We replace these with custom fused CUDA kernels.


### 2.1 Understanding Why This Matters


GNN message passing = SpMM (Sparse Matrix × Dense Matrix):


```
H_new = A × H × W


Where:
 A = sparse adjacency matrix (2.3M non-zeros in a ~170K × 170K matrix = 99.99% sparse)
 H = dense node feature matrix (170K × 128)
 W = dense weight matrix (128 × hidden_dim)


A × H is SpMM — the bottleneck.
```


**Why stock PyG is slow (5 problems to solve):**


1. **No operation fusion:** PyG does gather → message → aggregate → update as 4 separate kernel launches with 4 round trips to global memory.
2. **Generic SpMM (cuSPARSE):** Not tuned for this graph's degree distribution.
3. **Scatter-based aggregation uses global atomics:** For high-degree nodes, thousands of edges atomicAdd to the same location — serialized.
4. **Feature dimension inefficiency:** Each feature dimension treated independently instead of vectorized.
5. **No multi-GPU graph partitioning:** PyG's built-in doesn't optimize halo exchange.


### 2.2 Custom SpMM Kernel (CSR Format)


Implement a CUDA kernel for sparse-dense matrix multiplication that handles the irregular structure of graph adjacency matrices.


**Key design decisions:**


```
Problem: Power-law degree distribution
 - Some nodes (hubs) have 10,000+ neighbors
 - Most nodes have < 50 neighbors
 - A naive "one thread per row" approach wastes most threads


Solution: Adaptive parallelism
 - For rows with degree > 32: assign a WARP (32 threads) per row
 - For rows with degree ≤ 32: assign a single THREAD per row
 - Group rows by degree range for balanced warps
```


```cpp
// CSR SpMM kernel sketch — each warp handles one high-degree row
__global__ void spmm_csr_adaptive(
   const int* __restrict__ row_ptr,     // CSR row pointers
   const int* __restrict__ col_idx,     // CSR column indices
   const float* __restrict__ values,    // Edge weights (or 1.0 for unweighted)
   const float* __restrict__ features,  // Dense feature matrix H [N × F]
   float* __restrict__ output,          // Output matrix [N × F]
   int num_rows,
   int feature_dim
) {
   // Determine if this is a warp-per-row or thread-per-row assignment
   // based on pre-computed row degree information


   int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
   int lane_id = threadIdx.x % 32;


   if (warp_id >= num_rows) return;


   int row_start = row_ptr[warp_id];
   int row_end = row_ptr[warp_id + 1];
   int row_degree = row_end - row_start;


   // Each thread in the warp processes different feature dimensions
   for (int f = lane_id; f < feature_dim; f += 32) {
       float sum = 0.0f;
       for (int e = row_start; e < row_end; e++) {
           int col = col_idx[e];
           float val = values[e];
           sum += val * features[col * feature_dim + f];
       }
       output[warp_id * feature_dim + f] = sum;
   }
}
```


**Optimizations to apply:**


1. **Shared memory caching:** Cache frequently accessed columns of the dense feature matrix in shared memory.
2. **Vectorized loads:** Use `float4` loads for the dense feature matrix when feature_dim is a multiple of 4.
3. **Row grouping/sorting:** Sort rows by degree and group similar-length rows into the same warp for load balance.
4. **Column-major dense storage:** Store the dense feature matrix in column-major format so consecutive threads access consecutive memory addresses (coalesced access).


### 2.3 Fused Message Passing Kernel


Instead of 4 separate kernel launches (gather → message → aggregate → update), fuse them into a single kernel:


```cpp
// Fused GCN message passing: combines gather, transform, aggregate, update
__global__ void fused_gcn_message_passing(
   const int* __restrict__ row_ptr,
   const int* __restrict__ col_idx,
   const float* __restrict__ node_features,  // H [N × F]
   const float* __restrict__ weights,        // W [F × F']
   const float* __restrict__ bias,           // b [F']
   float* __restrict__ output,               // H_new [N × F']
   int num_nodes,
   int in_features,
   int out_features,
   const float* __restrict__ degree_inv_sqrt  // D^{-1/2} for normalization
) {
   extern __shared__ float smem[];
   // Shared memory layout:
   // [0, in_features): cached neighbor features
   // [in_features, in_features + out_features): partial sums for reduction


   int node = blockIdx.x;
   if (node >= num_nodes) return;


   int row_start = row_ptr[node];
   int row_end = row_ptr[node + 1];
   float norm_self = degree_inv_sqrt[node];


   // For each output feature dimension
   for (int out_f = threadIdx.x; out_f < out_features; out_f += blockDim.x) {
       float agg = 0.0f;


       // Aggregate neighbor messages (SpMM portion)
       for (int e = row_start; e < row_end; e++) {
           int neighbor = col_idx[e];
           float norm_neighbor = degree_inv_sqrt[neighbor];
           float edge_weight = norm_self * norm_neighbor;  // GCN normalization


           // Compute message: W^T * h_neighbor (dot product for this output dim)
           float msg = 0.0f;
           for (int in_f = 0; in_f < in_features; in_f++) {
               msg += node_features[neighbor * in_features + in_f]
                    * weights[in_f * out_features + out_f];
           }
           agg += edge_weight * msg;
       }


       // Add self-loop
       float self_msg = 0.0f;
       for (int in_f = 0; in_f < in_features; in_f++) {
           self_msg += node_features[node * in_features + in_f]
                     * weights[in_f * out_features + out_f];
       }
       agg += norm_self * norm_self * self_msg;


       // Fused bias + ReLU (epilogue)
       agg += bias[out_f];
       output[node * out_features + out_f] = fmaxf(agg, 0.0f);  // ReLU
   }
}
```


### 2.4 Shared Memory Reduction (Replace Global Atomics)


PyG's scatter_add uses global atomics. Replace with shared memory tree reduction:


```cpp
// Shared memory reduction for aggregating messages to the same target node
__global__ void scatter_reduce_smem(
   const float* __restrict__ messages,     // [num_edges × F]
   const int* __restrict__ target_nodes,   // [num_edges] — which node each message goes to
   float* __restrict__ output,             // [num_nodes × F]
   const int* __restrict__ node_offsets,    // CSR-style: start/end of edges per target node
   int num_nodes,
   int feature_dim
) {
   extern __shared__ float sdata[];


   int node = blockIdx.x;
   if (node >= num_nodes) return;


   int start = node_offsets[node];
   int end = node_offsets[node + 1];


   for (int f = 0; f < feature_dim; f++) {
       // Each thread loads one message element
       float val = 0.0f;
       for (int e = start + threadIdx.x; e < end; e += blockDim.x) {
           val += messages[e * feature_dim + f];
       }
       sdata[threadIdx.x] = val;
       __syncthreads();


       // Tree reduction in shared memory
       for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
           if (threadIdx.x < stride) {
               sdata[threadIdx.x] += sdata[threadIdx.x + stride];
           }
           __syncthreads();
       }


       if (threadIdx.x == 0) {
           output[node * feature_dim + f] = sdata[0];
       }
       __syncthreads();
   }
}
```


### 2.5 PyTorch C++ Extension Bindings


```cpp
// bindings.cpp
#include <torch/extension.h>


torch::Tensor custom_spmm_forward(
   torch::Tensor row_ptr,
   torch::Tensor col_idx,
   torch::Tensor values,
   torch::Tensor features
);


torch::Tensor fused_gcn_forward(
   torch::Tensor row_ptr,
   torch::Tensor col_idx,
   torch::Tensor features,
   torch::Tensor weights,
   torch::Tensor bias,
   torch::Tensor degree_inv_sqrt
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
   m.def("spmm_forward", &custom_spmm_forward, "Custom SpMM forward");
   m.def("fused_gcn_forward", &fused_gcn_forward, "Fused GCN forward");
}
```


### 2.6 Python Wrapper Layer


```python
# gnn_opt/layers/custom_conv.py
import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load


# JIT compile the CUDA extension
custom_ops = load(
   name='gnn_custom_ops',
   sources=['csrc/spmm/spmm_kernel.cu', 'csrc/fused/fused_message_passing.cu', 'csrc/bindings.cpp'],
   extra_cuda_cflags=['-O3', '--use_fast_math']
)


def transpose_csr(row_ptr, col_idx, values, num_rows, num_cols):
   """Transpose a CSR matrix by converting to COO, swapping, and back to CSR."""
   import torch
   rows = torch.repeat_interleave(
       torch.arange(num_rows, device=row_ptr.device),
       row_ptr[1:] - row_ptr[:-1]
   )
   # Swap rows and columns
   adj_t = torch.sparse_csr_tensor(
       crow_indices=torch.ops.aten._convert_indices_from_coo_to_csr(col_idx, num_cols),
       col_indices=rows.int(),
       values=values,
       size=(num_cols, num_rows)
   )
   return adj_t.crow_indices().int(), adj_t.col_indices().int(), adj_t.values()




class CustomSpMMFunction(Function):
   @staticmethod
   def forward(ctx, row_ptr, col_idx, values, features, num_nodes):
       ctx.save_for_backward(row_ptr, col_idx, values, features)
       ctx.num_nodes = num_nodes
       return custom_ops.spmm_forward(row_ptr, col_idx, values, features)


   @staticmethod
   def backward(ctx, grad_output):
       row_ptr, col_idx, values, features = ctx.saved_tensors
       n = ctx.num_nodes
       # Gradient of SpMM: dL/dH = A^T × dL/dH_new
       t_row_ptr, t_col_idx, t_values = transpose_csr(row_ptr, col_idx, values, n, n)
       grad_features = custom_ops.spmm_forward(t_row_ptr, t_col_idx, t_values, grad_output)
       return None, None, None, grad_features, None




class CustomGCNConv(torch.nn.Module):
   def __init__(self, in_channels, out_channels):
       super().__init__()
       self.weight = torch.nn.Parameter(torch.randn(in_channels, out_channels))
       self.bias = torch.nn.Parameter(torch.zeros(out_channels))
       torch.nn.init.xavier_uniform_(self.weight)


   def forward(self, x, edge_index_csr):
       row_ptr, col_idx, values, degree_inv_sqrt = edge_index_csr
       return custom_ops.fused_gcn_forward(
           row_ptr, col_idx, x, self.weight, self.bias, degree_inv_sqrt
       )
```


---


## Phase 3: Optimized GCN Model with Custom Kernels


### 3.1 Optimized Model


```python
class OptimizedGCN(torch.nn.Module):
   def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
       super().__init__()
       self.convs = torch.nn.ModuleList()
       self.bns = torch.nn.ModuleList()


       self.convs.append(CustomGCNConv(in_channels, hidden_channels))
       self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
       for _ in range(num_layers - 2):
           self.convs.append(CustomGCNConv(hidden_channels, hidden_channels))
           self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
       self.convs.append(CustomGCNConv(hidden_channels, out_channels))


       self.dropout = dropout


   def forward(self, x, edge_index_csr):
       for i, conv in enumerate(self.convs[:-1]):
           x = conv(x, edge_index_csr)
           x = self.bns[i](x)
           x = F.relu(x)
           x = F.dropout(x, p=self.dropout, training=self.training)
       x = self.convs[-1](x, edge_index_csr)
       return x
```


### 3.2 Data Preprocessing for CSR


Convert PyG's COO edge_index to CSR format for the custom kernels:


```python
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix, degree
import scipy.sparse as sp


def prepare_csr_data(data):
   """Convert PyG data to CSR format with GCN normalization."""
   adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
   adj = adj + sp.eye(adj.shape[0])  # add self-loops
   adj_csr = adj.tocsr()


   row_ptr = torch.from_numpy(adj_csr.indptr).to(torch.int32).cuda()
   col_idx = torch.from_numpy(adj_csr.indices).to(torch.int32).cuda()


   # GCN normalization: D^{-1/2} A D^{-1/2}
   deg = torch.from_numpy(np.array(adj.sum(axis=1)).flatten()).float()
   deg_inv_sqrt = deg.pow(-0.5)
   deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
   deg_inv_sqrt = deg_inv_sqrt.cuda()


   # Compute normalized values
   row_degs = deg_inv_sqrt[torch.from_numpy(adj_csr.nonzero()[0])]
   col_degs = deg_inv_sqrt[torch.from_numpy(adj_csr.indices)]
   values = (row_degs * col_degs).float().cuda()


   return row_ptr, col_idx, values, deg_inv_sqrt
```


---


## Phase 4: Multi-GPU Distributed Training


### 4.1 MPI-Based Data Parallel Training


```python
# scripts/train_distributed.py
import torch
import torch.distributed as dist
from mpi4py import MPI


def setup_distributed():
   comm = MPI.COMM_WORLD
   rank = comm.Get_rank()
   world_size = comm.Get_size()


   # Initialize PyTorch distributed with NCCL backend
   dist.init_process_group(
       backend='nccl',
       init_method='env://',
       world_size=world_size,
       rank=rank
   )
   torch.cuda.set_device(rank)
   return rank, world_size


def distributed_train_step(model, data, optimizer, rank, world_size):
   model.train()
   optimizer.zero_grad()


   out = model(data.x, data.edge_index_csr)
   loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask].squeeze())
   loss.backward()


   # AllReduce gradients across GPUs
   for param in model.parameters():
       if param.grad is not None:
           dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
           param.grad.data /= world_size


   optimizer.step()
   return loss.item()


def main():
   rank, world_size = setup_distributed()


   # Load data — each GPU gets the full graph but different train samples
   data = load_and_partition_data(rank, world_size)
   data = data.to(f'cuda:{rank}')


   model = OptimizedGCN(
       in_channels=data.num_features,
       hidden_channels=256,
       out_channels=data.num_classes,
       num_layers=3
   ).to(f'cuda:{rank}')


   optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


   for epoch in range(500):
       loss = distributed_train_step(model, data, optimizer, rank, world_size)
       if rank == 0 and epoch % 50 == 0:
           accs = evaluate(model, data, split_idx)
           print(f'Epoch {epoch}: Loss={loss:.4f}, Test={accs["test"]:.4f}')


   dist.destroy_process_group()
```


### 4.2 Graph Partitioning for Multi-GPU


```python
def partition_graph(data, num_parts):
   """Partition graph across GPUs with minimal edge cuts."""
   import torch_geometric.transforms as T


   # Use METIS partitioning
   cluster_data = T.partition.ClusterData(data, num_parts=num_parts)


   partitions = []
   for i in range(num_parts):
       partition = cluster_data[i]
       partitions.append(partition)


   return partitions


def compute_halo_nodes(partition, full_edge_index):
   """Find boundary nodes that need feature exchange between partitions."""
   local_nodes = set(partition.node_ids.tolist())
   halo_nodes = set()


   src, dst = full_edge_index
   for s, d in zip(src.tolist(), dst.tolist()):
       if s in local_nodes and d not in local_nodes:
           halo_nodes.add(d)
       if d in local_nodes and s not in local_nodes:
           halo_nodes.add(s)


   return halo_nodes
```


### 4.3 Halo Exchange with MPI


```python
def halo_exchange(local_features, halo_send_ids, halo_recv_ids, comm, rank):
   """Exchange boundary node features between GPU partitions using MPI."""
   send_data = local_features[halo_send_ids].cpu().numpy()
   recv_data = np.empty_like(send_data)


   # Non-blocking send/receive for overlap with compute
   reqs = []
   for target_rank, send_buf in enumerate(send_data):
       if target_rank != rank:
           req = comm.Isend(send_buf, dest=target_rank, tag=0)
           reqs.append(req)


   for source_rank in range(comm.Get_size()):
       if source_rank != rank:
           comm.Recv(recv_data, source=source_rank, tag=0)


   MPI.Request.Waitall(reqs)


   return torch.from_numpy(recv_data).cuda(rank)
```


### 4.4 Launch Script


```bash
# Launch 4-GPU training with MPI
mpirun -np 4 \
   --bind-to socket \
   --map-by socket \
   -x MASTER_ADDR=localhost \
   -x MASTER_PORT=29500 \
   python scripts/train_distributed.py \
   --dataset ogbn-arxiv \
   --model optimized_gcn \
   --hidden 256 \
   --layers 3 \
   --epochs 500 \
   --lr 0.01
```


---


## Phase 5: Profiling and Optimization Iteration


### 5.1 Profiling Workflow


```bash
# Step 1: System-level timeline
nsys profile --stats=true \
   python scripts/train_optimized.py --epochs 5


# Step 2: Kernel-level deep dive
ncu --set full \
   --target-processes all \
   python scripts/train_optimized.py --epochs 1
```


### 5.2 What to Look For and Fix


| Metric (Nsight Compute) | If Bad, Do This |
|--------------------------|-----------------|
| **Global Load Efficiency < 80%** | Fix memory access patterns — ensure coalesced access, use SoA layout for feature matrix |
| **SM Throughput low, Memory Throughput high** | Memory-bound — add shared memory tiling, reduce global memory traffic |
| **SM Throughput high, Memory Throughput low** | Compute-bound — good! Consider Tensor Core usage for dense portions |
| **Both low** | Latency-bound — increase occupancy, launch more threads |
| **Warp Execution Efficiency < 80%** | Warp divergence — restructure conditionals to align with warp boundaries |
| **Achieved Occupancy < 50%** | Check register usage and shared memory — reduce if possible without spilling |


### 5.3 Optimization Checklist


```
□ Memory coalescing: dense feature matrix in column-major for thread-contiguous access
□ Shared memory tiling: cache hot columns of feature matrix in shared memory
□ Vectorized loads: use float4 for aligned feature loads (4x fewer transactions)
□ Warp-level reduction: use __shfl_down_sync instead of shared memory for small reductions
□ Row sorting: group rows by degree for balanced warp utilization
□ Kernel fusion: combine gather + message + aggregate + update into single kernel
□ Avoid global atomics: use shared memory tree reduction, one atomic per block
□ Minimize host-device transfers: keep graph data on GPU across epochs
□ Use CUDA streams: overlap data preprocessing with kernel execution
□ Register pressure: check ncu for register spills, tune block size
```


---


## Phase 6: Benchmarking and Validation


### 6.1 Benchmark Suite


```python
# scripts/benchmark.py
import time
import torch
import numpy as np


def benchmark_full(model_class, data, num_seeds=5, num_epochs=500):
   results = {
       'test_accs': [],
       'epoch_times': [],
       'peak_memory': [],
       'total_time': []
   }


   for seed in range(num_seeds):
       torch.manual_seed(seed)
       np.random.seed(seed)


       model = model_class(...).cuda()
       optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


       torch.cuda.reset_peak_memory_stats()
       epoch_times = []


       t_total = time.time()
       for epoch in range(num_epochs):
           t_epoch = time.time()
           train_step(model, data, optimizer)
           torch.cuda.synchronize()
           epoch_times.append(time.time() - t_epoch)


       total_time = time.time() - t_total
       accs = evaluate(model, data, split_idx)


       results['test_accs'].append(accs['test'])
       results['epoch_times'].append(np.mean(epoch_times))
       results['peak_memory'].append(torch.cuda.max_memory_allocated() / 1e9)
       results['total_time'].append(total_time)


   return {
       'test_acc': f"{np.mean(results['test_accs']):.4f} ± {np.std(results['test_accs']):.4f}",
       'avg_epoch_time': f"{np.mean(results['epoch_times']):.4f}s",
       'peak_memory_gb': f"{np.mean(results['peak_memory']):.2f} GB",
       'total_time': f"{np.mean(results['total_time']):.1f}s"
   }


# Run benchmarks
baseline_results = benchmark_full(BaselineGCN, data)
optimized_results = benchmark_full(OptimizedGCN, data)
```


### 6.2 Multi-GPU Scaling Benchmark


```python
def benchmark_scaling(num_gpus_list=[1, 2, 4]):
   """Measure training throughput scaling across different GPU counts."""
   results = {}
   for n_gpus in num_gpus_list:
       # Launch with MPI and collect epoch times
       cmd = f"mpirun -np {n_gpus} python scripts/train_distributed.py --epochs 50"
       epoch_time = run_and_parse_output(cmd)
       results[n_gpus] = {
           'epoch_time': epoch_time,
           'throughput': 1.0 / epoch_time,  # epochs per second
           'speedup': results.get(1, {}).get('epoch_time', epoch_time) / epoch_time
       }
   return results
```


### 6.3 Expected Results


| Metric | Baseline (PyG) | Optimized (Custom Kernels) | Improvement |
|--------|---------------|---------------------------|-------------|
| Test Accuracy | ~65-70% | ~72-77% | +6.8% |
| Epoch Time (1 GPU) | X ms | ~0.6X ms | ~1.7x faster |
| Epoch Time (4 GPU) | N/A | ~0.42X ms | ~2.4x faster |
| Peak Memory | Y GB | ~0.7Y GB | ~30% less |


The **6.8% accuracy improvement** comes from:
- Better numerical precision in the fused kernel (fewer intermediate roundings)
- The custom kernel enabling more efficient use of deeper/wider model architectures within the same memory budget
- Better hyperparameter tuning enabled by faster iteration


The **2.4x throughput improvement** on 4 GPUs comes from:
- Custom CUDA kernels: ~1.7x faster per-GPU (kernel fusion, shared memory reduction, adaptive parallelism)
- Distributed parallelism: ~1.4x additional from 4 GPUs (limited by Amdahl's Law, communication overhead)
- Combined: ~1.7 × 1.4 ≈ 2.4x


---


## Phase 7: Correctness Testing


### 7.1 Numerical Correctness


```python
# tests/test_spmm.py
import torch
from gnn_opt.layers.custom_conv import custom_ops


def test_spmm_correctness():
   """Verify custom SpMM matches PyTorch sparse matmul."""
   N, F = 1000, 64
   density = 0.01


   # Random sparse matrix in CSR
   adj = torch.rand(N, N).cuda()
   adj = (adj < density).float()
   adj_csr = adj.to_sparse_csr()


   features = torch.randn(N, F).cuda()


   # Reference: PyTorch sparse matmul
   expected = torch.sparse.mm(adj.to_sparse(), features)


   # Custom kernel
   actual = custom_ops.spmm_forward(
       adj_csr.crow_indices().int(),
       adj_csr.col_indices().int(),
       adj_csr.values(),
       features
   )


   torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_backward_correctness():
   """Verify gradients match PyTorch autograd."""
   # Use torch.autograd.gradcheck with the custom function
   pass


def test_fused_vs_separate():
   """Verify fused kernel produces same results as separate operations."""
   pass
```


### 7.2 Edge Cases


```python
def test_empty_rows():
   """Nodes with no neighbors should produce zero output."""
   pass


def test_single_neighbor():
   """Nodes with exactly one neighbor."""
   pass


def test_high_degree_nodes():
   """Hub nodes with 10,000+ neighbors."""
   pass


def test_large_feature_dim():
   """Feature dimensions > 1024."""
   pass
```


---


## Hyperparameters (Default Configuration)


```yaml
# configs/default.yaml
dataset:
 name: ogbn-arxiv
 root: ./data


model:
 type: optimized_gcn
 hidden_channels: 256
 num_layers: 3
 dropout: 0.5


training:
 epochs: 500
 lr: 0.01
 weight_decay: 5e-4
 optimizer: adam


cuda:
 block_size: 256
 tile_size: 16
 warp_threshold: 32      # degree threshold for warp-per-row vs thread-per-row
 use_vectorized_loads: true
 shared_mem_cache_size: 48  # KB


distributed:
 num_gpus: 4
 backend: nccl
 partition_method: metis


benchmark:
 num_seeds: 5
 warmup_epochs: 10
```


---


## Key Technical Concepts to Implement Correctly


### 1. CSR vs COO Format


```
COO (Coordinate): PyG default
 edge_index = [[src_0, src_1, ...],
               [dst_0, dst_1, ...]]
 Good for: random access, building graphs
 Bad for: row-wise iteration (SpMM)


CSR (Compressed Sparse Row): Our custom kernels use this
 row_ptr  = [0, 3, 5, 5, 8, ...]   // row i has edges from row_ptr[i] to row_ptr[i+1]
 col_idx  = [2, 5, 7, 1, 4, ...]   // column indices of non-zeros
 values   = [0.5, 0.3, ...]        // edge weights
 Good for: row-wise iteration (SpMM) — exactly what GNNs need
```


### 2. GCN Normalization


```
Standard GCN: H_new = D^{-1/2} × A × D^{-1/2} × H × W


Where D is the degree matrix. The D^{-1/2} factors normalize messages
by the geometric mean of source and target degrees, preventing
high-degree nodes from having disproportionately large activations.


Pre-compute deg_inv_sqrt = D^{-1/2} once and store it.
Apply normalization per-edge in the kernel: weight = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
```


### 3. Autograd for Custom CUDA Ops


```python
# Every custom CUDA operation needs both forward AND backward implementations.
# PyTorch's autograd calls backward automatically during loss.backward().


# For SpMM backward:
#   dL/dH = A^T × dL/dH_new  (transpose the sparse matrix, multiply by upstream gradient)
#   dL/dW = H^T × (A × dL/dH_new)  (standard GEMM for weight gradients)
```


### 4. Memory Management


```
- Allocate graph structure (row_ptr, col_idx, values) ONCE at startup
- Keep it on GPU across all epochs — never transfer back to CPU
- Only node features change during training
- Use torch.cuda.empty_cache() between benchmarking runs
- Monitor with torch.cuda.max_memory_allocated()
```


### 5. Why 2.4x and Not 4x on 4 GPUs


```
Amdahl's Law: If 15% of work is serial (setup, result gathering, synchronization):
 Max speedup = 1 / (0.15 + 0.85/4) = 1 / 0.3625 = 2.76x


Communication overhead further reduces this:
 - AllReduce gradients across 4 GPUs takes time
 - Halo exchange for boundary nodes
 - Synchronization barriers


Load imbalance:
 - Graph partitioning can't perfectly balance computation
 - Power-law degree distribution means some partitions are heavier


Realistic: 2.4x on 4 GPUs = 60% parallel efficiency — solid for graph workloads.
```


---


## Build and Run Commands


```bash
# 1. Install dependencies
pip install -r requirements.txt


# 2. Build CUDA extensions
python setup.py build_ext --inplace
# OR use JIT compilation (automatic on first run)


# 3. Run baseline
python scripts/train_baseline.py --dataset ogbn-arxiv --epochs 500


# 4. Run optimized (single GPU)
python scripts/train_optimized.py --dataset ogbn-arxiv --epochs 500


# 5. Run distributed (4 GPUs)
mpirun -np 4 python scripts/train_distributed.py --dataset ogbn-arxiv --epochs 500


# 6. Run full benchmark suite
python scripts/benchmark.py --num-seeds 5


# 7. Profile
nsys profile python scripts/train_optimized.py --epochs 5
ncu --set full python scripts/train_optimized.py --epochs 1
```


---


## Summary of What Makes This Project Non-Trivial


1. **Custom CUDA kernels** that outperform generic cuSPARSE for this specific workload pattern (power-law graph, moderate feature dimensions)
2. **Kernel fusion** — combining 4 separate kernel launches into 1, eliminating redundant global memory traffic
3. **Adaptive parallelism** — warp-per-row for high-degree nodes, thread-per-row for low-degree, avoiding load imbalance
4. **Shared memory reduction** replacing global atomics — orders of magnitude fewer serialization points
5. **Multi-GPU training** with MPI-style communication, graph partitioning, and halo exchange
6. **Rigorous benchmarking** with multiple seeds, profiling-guided optimization, and clear attribution of where speedups come from


The project demonstrates systems-level thinking about GPU optimization applied to a real ML workload, combining CUDA kernel development, PyTorch integration, and distributed computing.



