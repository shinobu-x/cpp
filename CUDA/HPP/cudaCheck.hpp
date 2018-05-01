#ifndef CUDACHECK_HPP
#define CUDACHECK_HPP

#include <cuda_runtime.h>
#include <sstream>

inline void cudaCheck(cudaError_t status) {
  if (status != cudaSuccess) {
    std::stringstream error;
    error << cudaGetErrorString(status) << '\n';
  }
}
#endif // CUDACHECK_HPP
