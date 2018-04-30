#ifndef CUDASETUPDEVICE_HPP
#define CUDASETUPDEVICE_HPP

#include <cuda_runtime.h>
#include <iostream>

void cudaSetupDevice() {
  int dev = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);
  std::cout << "Device ID: " << dev << '\n';
  std::cout << "Device Name: " << prop.name << '\n';
  cudaSetDevice(dev);
}

#endif
