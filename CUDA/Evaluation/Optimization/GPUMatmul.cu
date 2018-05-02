#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <CUDA/HPP/CalElapsedTime.hpp>

typedef float value_type;
int N = std::pow(2, 10);
std::size_t NBytes = N * N * sizeof(value_type);

value_type* h_a;
value_type* h_b;
value_type* h_c;
value_type* d_a;
value_type* d_b;
value_type* d_c;
int i, j, k;

void Init() {
  h_a = (value_type*)malloc(NBytes);
  h_b = (value_type*)malloc(NBytes);
  h_c = (value_type*)malloc(NBytes);

  cudaMalloc((void**)&d_a, NBytes);
  cudaMalloc((void**)&d_b, NBytes);
  cudaMalloc((void**)&d_c, NBytes);

  for (k = 0; k < N; ++k) {
    for (i = 0; i < N; ++i) {
      h_a[k * N + i] = (value_type)(k + 1) * 0.1f;
    }
  }

  for (j = 0; j < N; ++j) {
    for (k = 0; k < N; ++k) {
      h_b[j * N + k] = (value_type)(j + 1) * 0.1f;
    }
  }

  for (j = 0; j < N; ++j) {
    for (i = 0; i < N; ++i) {
      h_c[j * N + i] = 0.0f;
    }
  }
}

__global__
void Matmul(value_type* a, value_type* b, value_type* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  for (int k = 0; k < n; ++k) {
    c[i * n + j] += a[i * n + k] * b[k * n + j];
  } 
}

void SpawnKernel() {
  Matmul<<<N / 256, 256>>>(d_a, d_b, d_c, N);
}

auto main() -> decltype(0) {
  Init();

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(h_a);
  free(h_b);
  free(h_c);
  return 0;
}
