#include <cstdlib>
#include <cuda_runtime.h>
#include <mpi.h>

#include <CUDA/HPP/InitData.hpp>
#include <CUDA/HPP/cudaSetupDevice.hpp>
#include <CUDA/HPP/sumArraysOnDevice.hpp>
void DoIt() {
  cudaSetupDevice();

  typedef float value_type;
  typedef float* ptr_type;
  int N = 1 << 24;
  std::size_t NBytes = N * sizeof(value_type);

  ptr_type h_a;
  ptr_type h_b;
  ptr_type h_c;
  h_a = (ptr_type)malloc(NBytes);
  h_b = (ptr_type)malloc(NBytes);
  h_c = (ptr_type)malloc(NBytes);

  InitData(h_a, N);
  InitData(h_b, N);
  InitData(h_c, N);

  ptr_type d_a;
  ptr_type d_b;
  ptr_type d_c;
  cudaMalloc(&d_a, NBytes);
  cudaMalloc(&d_b, NBytes);
  cudaMalloc(&d_c, NBytes);

  cudaMemcpy(d_a, h_a, NBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, NBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c, NBytes, cudaMemcpyHostToDevice);

  dim3 Block(N);
  dim3 Grid((N * Block.x - 1) / Block.x);
  // sumArraysOnDevice<<<Grid, Block>>>(d_a, d_b, d_c, N);
  cudaDeviceSynchronize();

  cudaMemcpy(h_c, d_c, NBytes, cudaMemcpyDeviceToHost);

  free(h_a);
  free(h_b);
  free(h_c);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

auto main(int argc, char** argv) -> decltype(0) {
  MPI_Init(&argc, &argv);

  DoIt();

  MPI_Finalize();
  return 0;
}
