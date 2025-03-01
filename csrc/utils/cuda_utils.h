// csrc/utils/cuda_utils.h
#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                  \
   do {                                                                  \
       cudaError_t err = call;                                           \
       if (err != cudaSuccess) {                                         \
           fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                   __FILE__, __LINE__, cudaGetErrorString(err));          \
           exit(EXIT_FAILURE);                                           \
       }                                                                 \
   } while (0)

#define CUDA_CHECK_LAST_ERROR()                                           \
   do {                                                                  \
       cudaError_t err = cudaGetLastError();                              \
       if (err != cudaSuccess) {                                         \
           fprintf(stderr, "CUDA kernel error at %s:%d: %s\n",            \
                   __FILE__, __LINE__, cudaGetErrorString(err));          \
           exit(EXIT_FAILURE);                                           \
       }                                                                 \
   } while (0)

inline int div_ceil(int a, int b) { return (a + b - 1) / b; }
