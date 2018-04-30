#ifndef CUDACHECK_HPP
#define CUDACHECK_HPP

#include <cuda_runtime.h>
#include <sstream>

inline void cudaCheck(const char* expr, cudaError_t status, const char* file,
  int line) {
  if (status != cudaSuccess) {
    std::stringstream error;
    error << cudaGetErrorString(status) << "\n\t" << expr << " at " << file
      << '.' << line << std::endl;
  }
}
#endif // CUDACHECK_HPP
