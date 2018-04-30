#include <cstdlib>

#include "../HPP/sumArraysOnDevice.hpp"
#include "../HPP/cudaSetupDevice.hpp"

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

  // Addresses reservations for Device
  value_type* d_a;
  value_type* d_b;
  value_type* d_c;
  cudaMalloc((value_type**)&d_a, nbytes);
  cudaMalloc((value_type**)&d_b, nbytes);
  cudaMalloc((value_type**)&d_c, nbytes);

  // Copy  
    
  return 0;
}
