#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

#include "../HPP/cudaCheck.hpp"

void printMatrix(int* c, const int nx, const int ny) {
  int* ic = c;
  std::cout << "Matrix: (" << nx << "," << ny << ")\n";

  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      std::cout << ic[ix] << "\n";
    }
    ic += nx;
    std::cout << "\n";
  }

  std::cout << "\n";
}

__global__
void printThreadIndex(int* a, const int nx, const int ny) {
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;
  std::size_t idx = iy * nx + ix;

  std::cout << "Thread ID: (" << threadIdx.x << ", " << threadIdx.y << ")\n";
  std::cout << "Block ID: (" << blockIdx.x << ", " << blockIdx.y << ")\n";
  std::cout << "Coordinate: (" << ix << ", " << iy << ")\n";
  std::cout << "Global Index: " << idx << "\n";
  std::cout << "Value: " << a[idx] << "\n";
}

auto main() -> decltype(0) {
  int dev = 0;
  cudaDeviceProp prop;
  cudaCheck(cudaGetDeviceProperties(&prop, dev));
  std::cout << "Device: " << dev << " / " << prop.name << "\n";
  cudaCheck(cudaSetDevice(dev));

  int nx = 8;
  int ny = 6;
