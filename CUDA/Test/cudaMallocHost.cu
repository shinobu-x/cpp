#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>

#include "../Hpp/CalElapsedTime.hpp"

__global__
void CalSine(const float* angle, float* sine, std::size_t size) {
  int thread = blockDim.x * blockIdx.x + threadIdx.x;
  if (thread < size) {
    sine[thread] = sinf(angle[thread]);
  }
}

void DoIt() {
  typedef float value_type;
  const std::size_t n = 100000000;
  std::size_t data_size = n * sizeof(float);

  // Hostでアドレス確保
  float* h_angle;
  float* h_sine;

  cudaMallocHost(&h_angle, data_size);
  cudaMallocHost(&h_sine, data_size);

  // Deviceでアドレス確保
  float* d_angle;
  float* d_sine;
  cudaMalloc(&d_angle, data_size);
  cudaMalloc(&d_sine, data_size);

  // データ転送: Host -> Device
  // cudaMemcpyAsync(dest, src, size, type)
  cudaMemcpyAsync(d_angle, h_angle, data_size, cudaMemcpyDefault);
  // Kernel起動
  CalSine<<< (n + 255) / 256, 256 >>>(d_angle, d_sine, n);
  // データ転送: Device -> Host
  cudaMemcpyAsync(h_angle, d_angle, data_size, cudaMemcpyDefault);

  // 処理完了まで待機
  cudaStreamSynchronize(nullptr);

  cudaFree(d_angle);
  cudaFree(d_sine);

  cudaFree(h_angle);
  cudaFree(h_sine);

  cudaDeviceReset();
}

auto main() -> decltype(0) {
  std::cout << CalElapsedTime<>::Execution(DoIt) << '\n';
  return 0;
} 
