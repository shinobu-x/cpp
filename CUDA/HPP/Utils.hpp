#ifndef UTILS_HPP
#define UTILS_HPP

#include <iomanip>
#include <iostream>
#include <memory>

#include <cuda.h>

#define cudaCheck(f) {                                                         \
  cudaError_t status = (f);                                                    \
  if (status != cudaSuccess) {                                                 \
    std::cout << #f ": " << status << std::endl;                               \
    std::exit(1);                                                              \
  }                                                                            \
}

#define cudnnCheck(f) {                                                        \
  cudnnStatus_t status = (f);                                                  \
  if (status != CUDNN_STATUS_SUCCESS) {                                        \
    std::cout << #f ": " << status << std::endl;                               \
    std::exit(1);                                                              \
  }                                                                            \
}

template <typename T = void>
inline std::shared_ptr<T> MemAlloc(std::size_t s){
  typedef T value_type;
  typedef T* ptr_type;
  ptr_type ptr;
  cudaCheck(cudaMalloc(&ptr, s));
  return std::shared_ptr<value_type>(ptr, [](ptr_type ptr) {
    cudaFree(ptr); });
}

#endif
