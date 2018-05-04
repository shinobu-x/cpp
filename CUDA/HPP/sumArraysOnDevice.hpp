#ifndef SUMARRAYONDEVICE_HPP
#define SUMARRAYONDEVICE_HPP

#include <cuda_runtime.h>

__global__
void sumArraysOnDevice(float* a, float* b, float* c, const int thread_max) {
  int thread = blockDim.x * blockIdx.x + threadIdx.x;
  if (thread < thread_max) {
    c[thread] = a[thread] + b[thread];
  }
}

#endif
