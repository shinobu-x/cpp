#include <cstdlib>
#include <cuda_runtime.h>
#include <CUDA/HPP/InitData.hpp>
#include <CUDA/HPP/sumArraysOnDevice.hpp>

typedef float value_type;
int N = 1 << 24;
std::size_t NBytes = N * sizeof(value_type);

value_type* h_a;
value_type* h_b;
value_type* h_c;
value_type* d_a;
value_type* d_b;
value_type* d_c;

template <typename T>
void DoKernel(T* a, T* b, T* c) {
  dim3 Block(1024);
  dim3 Grid((N + Block.x - 1) / Block.x);
  sumArraysOnDevice<<<Grid, Block>>>(a, b, c, N);
}

void Malloc() {
  h_a = (value_type*)malloc(NBytes);
  h_b = (value_type*)malloc(NBytes);
  h_c = (value_type*)malloc(NBytes);
  InitData(h_a, N);
  InitData(h_b, N);
  InitData(h_c, N);

  cudaMalloc(&d_a, NBytes);
  cudaMalloc(&d_b, NBytes);
  cudaMalloc(&d_c, NBytes);

  cudaMemcpy(d_a, h_a, NBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, NBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c, NBytes, cudaMemcpyHostToDevice);

  DoKernel(d_a, d_b, d_c);

  cudaMemcpy(h_c, d_c, NBytes, cudaMemcpyDeviceToHost);
 
  cudaStreamSynchronize(nullptr);
  
  free(h_a);
  free(h_b);
  free(h_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

void MallocHost() {
  cudaMallocHost(&h_a, NBytes);
  cudaMallocHost(&h_b, NBytes);
  cudaMallocHost(&h_c, NBytes);
  InitData(h_a, N);
  InitData(h_b, N);
  InitData(h_c, N);

  cudaMalloc(&d_a, NBytes);
  cudaMalloc(&d_b, NBytes);
  cudaMalloc(&d_c, NBytes); 

  cudaMemcpyAsync(d_a, h_a, NBytes, cudaMemcpyDefault);
  cudaMemcpyAsync(d_b, h_b, NBytes, cudaMemcpyDefault);
  cudaMemcpyAsync(d_c, h_c, NBytes, cudaMemcpyDefault);

  DoKernel(d_a, d_b, d_c);

  cudaMemcpyAsync(h_c, d_c, NBytes, cudaMemcpyDefault);

  cudaStreamSynchronize(nullptr);

  cudaFree(h_a);
  cudaFree(h_b);
  cudaFree(h_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

auto main() -> decltype(0) {
//  Malloc();
//  MallocHost();
  return 0;
}
