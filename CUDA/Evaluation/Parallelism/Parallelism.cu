#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <CUDA/HPP/CalElapsedTime.hpp>
#include <CUDA/HPP/InitData.hpp>

typedef float value_type;
int N = std::pow(2, 20); 
std::size_t NBytes = N * sizeof(value_type); 

value_type* h_a;
value_type* h_b;
value_type* h_c;

value_type* d_a;
value_type* d_b;
value_type* d_c;

__global__
void DoCal(float* a, float* b, float* c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  c[i] = a[i] * b[i];
}

void SpawnKernel() {
  // Spawn Kernel
  // gridDim.x = N / 256
  // blockDim.x = 256 => 256 Threads
  int threads = 1024;
  DoCal<<<N / threads, threads>>>(d_a, d_b, d_c);
  cudaDeviceSynchronize();
}

// With cudaMemcpy
void _cudaMemcpy() {
  h_a = (value_type*)malloc(NBytes);
  h_b = (value_type*)malloc(NBytes);
  h_c = (value_type*)malloc(NBytes);

  InitData(h_a, N); 
  InitData(h_b, N);
  InitData(h_c, N);

  cudaMalloc((void**)&d_a, NBytes);
  cudaMalloc((void**)&d_b, NBytes);
  cudaMalloc((void**)&d_c, NBytes);

  // Copy from Host to Device
  cudaMemcpy(d_a, h_a, NBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, NBytes, cudaMemcpyHostToDevice);

  std::cout << CalElapsedTime<>::Execution(SpawnKernel) << "\n";

  cudaMemcpy(h_c, d_c, NBytes, cudaMemcpyDeviceToHost);

  // Free
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  free(h_a);
  free(h_b);
  free(h_c);
}

auto main() -> decltype(0) {
  std::size_t NBytes = N * sizeof(value_type); 
  std::cout << "A number of elements: " << N << "\n";
  std::cout << "A number of data size: " << NBytes << "\n";

  _cudaMemcpy();

  return 0;
}
