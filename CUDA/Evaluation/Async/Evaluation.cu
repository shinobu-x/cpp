// nvprof --print-gpu-trace ./a.out
#include <boost/thread/future.hpp>
#include <boost/thread.hpp>
#include <cstdlib>
#include <CUDA/HPP/InitData.hpp>
#include <CUDA/HPP/sumArraysOnDevice.hpp>

typedef float value_type;

void SpawnKernel(int threads, std::size_t nbytes, cudaStream_t stream) {
  value_type* h_a;
  value_type* h_b;
  value_type* h_c;
  h_a = (value_type*)malloc(nbytes);
  h_b = (value_type*)malloc(nbytes);
  h_c = (value_type*)malloc(nbytes);

  InitData(h_a, threads);
  InitData(h_b, threads);

  value_type* d_a;
  value_type* d_b;
  value_type* d_c;
  cudaMalloc((value_type**)&d_a, nbytes);
  cudaMalloc((value_type**)&d_b, nbytes);
  cudaMalloc((value_type**)&d_c, nbytes);

  cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, nbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c, nbytes, cudaMemcpyHostToDevice);

  dim3 Block(threads);
  dim3 Grid((threads + Block.x - 1) / Block.x);
  sumArraysOnDevice<<<Grid, Block, 0, stream>>>(
    d_a, d_b, d_c, threads);
  cudaMemcpyAsync(h_c, d_c, nbytes, cudaMemcpyDeviceToHost, stream);
  cudaDeviceSynchronize();

  free(h_a);
  free(h_b);
  free(h_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

void EvaluateStream() {
  const int N = 8;
  cudaStream_t s[N];
  value_type* Data[N];
  std::size_t NBytes = N * sizeof(value_type);

  for (int i = 0; i < N; ++i) {
    cudaStreamCreate(&s[i]);
    cudaMalloc(&Data[i], NBytes);
    SpawnKernel(N, NBytes, s[i]);
  }
}

auto main() -> decltype(0) {
  EvaluateStream();
  return 0;
}
