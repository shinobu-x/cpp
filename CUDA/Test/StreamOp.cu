#include <cstdlib>
#include <cuda_runtime.h>
#include <CUDA/HPP/InitData.hpp>
#include <CUDA/HPP/sumArraysOnDevice.hpp>

template <typename ValueType>
void DoIt() {
  typedef ValueType value_type;
  int N = 1 << 24;
  std::size_t NBytes = N * sizeof(value_type);

  value_type* h_a;
  value_type* h_b;
  value_type* h_c;
  value_type* d_a;
  value_type* d_b;
  value_type* d_c;

  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);

  cudaMalloc((void**)&d_a, NBytes);
  cudaMalloc((void**)&d_b, NBytes);
  cudaMalloc((void**)&d_c, NBytes);
  cudaHostAlloc((void**)&h_a, NBytes, cudaHostAllocDefault);
  cudaHostAlloc((void**)&h_b, NBytes, cudaHostAllocDefault);
  cudaHostAlloc((void**)&h_c, NBytes, cudaHostAllocDefault);

  InitData(h_a, N);
  InitData(h_b, N);

  cudaMemcpyAsync(d_a, h_a, NBytes, cudaMemcpyHostToDevice, s1);
  cudaMemcpyAsync(d_b, h_b, NBytes, cudaMemcpyHostToDevice, s2);

  dim3 Block(1024);
  dim3 Grid((N + Block.x - 1) / Block.x);
 
  sumArraysOnDevice<<<Grid, Block, 0, s1>>>(d_a, d_b, d_c, N);
  sumArraysOnDevice<<<Grid, Block, 0, s2>>>(d_a, d_b, d_c, N);

  cudaMemcpyAsync(h_c, d_c, NBytes, cudaMemcpyDeviceToHost, s1);
  cudaMemcpyAsync(h_c, d_c, NBytes, cudaMemcpyDeviceToHost, s2);

  cudaDeviceSynchronize();

  cudaFree(h_a);
  cudaFree(h_b);
  cudaFree(h_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

auto main() -> decltype(0) {
  DoIt<float>();

  return 0;
}
