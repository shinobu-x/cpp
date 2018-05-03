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
void MatmulBad(value_type* a, value_type* b, value_type* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  for (int k = 0; k < n; ++k) {
    c[j + i * n] += a[k + i * n] * b[j + k * n];
  }
}

__global__
void MatmulGood(value_type* a, value_type* b, value_type* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  for (int k = 0; k < n; ++k) {
    c[i + j * n] += a[i + k * n] * b[k + j * n];
  } 
}

__global__
void MatmulRegister(value_type* a, value_type* b, value_type* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  value_type tmp = 0.0f; // Register

  for (int k = 0; k < n; ++k) {
    tmp += a[k + i * n] * b[j + k * n];
  }

  c[i + j * n] = tmp;
}

void SpawnGoodKernel() {
  MatmulGood<<<N / 128, 128>>>(d_a, d_b, d_c, N);
  cudaDeviceSynchronize();
}

void SpawnBadKernel() {
  MatmulBad<<<N / 128, 128>>>(d_a, d_b, d_c, N);
  cudaDeviceSynchronize();
}

void SpawnRegisterKernel() {
  MatmulRegister<<<N / 128, 128>>>(d_a, d_b, d_c, N);
  cudaDeviceSynchronize();
}

auto main() -> decltype(0) {
  Init();

  std::cout << CalElapsedTime<>::Execution(SpawnGoodKernel) << "\n";
  std::cout << CalElapsedTime<>::Execution(SpawnBadKernel) << "\n";
  std::cout << CalElapsedTime<>::Execution(SpawnRegisterKernel) << "\n";

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(h_a);
  free(h_b);
  free(h_c);
  return 0;
}
