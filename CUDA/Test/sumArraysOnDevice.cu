#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

#include <CUDA/HPP/InitData.hpp>
#include <CUDA/HPP/cudaSetupDevice.hpp>
#include <CUDA/HPP/sumArraysOnDevice.hpp>


auto main() -> decltype(0) {
  cudaSetupDevice();

  typedef float value_type;
  int threads = 1 << 24;
  std::size_t nbytes = threads * sizeof(value_type);

  // Addresses reservations for Host
  value_type* h_a;
  value_type* h_b;
  value_type* h_c;
  h_a = (value_type*)malloc(nbytes);
  h_b = (value_type*)malloc(nbytes);
  h_c = (value_type*)malloc(nbytes);

  // Data initializations
  InitData(h_a, threads);
  InitData(h_b, threads);

  memset(h_c, 0, nbytes);

  // Addresses reservations for Device
  value_type* d_a;
  value_type* d_b;
  value_type* d_c;
  cudaMalloc((value_type**)&d_a, nbytes);
  cudaMalloc((value_type**)&d_b, nbytes);
  cudaMalloc((value_type**)&d_c, nbytes);

  // Copy data from host to device
  cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, nbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c, nbytes, cudaMemcpyHostToDevice);

  // Execute kernel
  dim3 block(1024);
  dim3 grid((threads + block.x - 1) / block.x);
  sumArraysOnDevice<<<grid, block>>>(d_a, d_b, d_c, threads);
  cudaDeviceSynchronize();

  // Copy data from device to host
  cudaMemcpy(h_c, d_c, nbytes, cudaMemcpyDeviceToHost);

  free(h_a);
  free(h_b);
  free(h_c);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
