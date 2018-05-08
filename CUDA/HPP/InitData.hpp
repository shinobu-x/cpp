#ifndef INITDATA_HPP
#define INITDATA_HPP

#include <cuda_runtime.h>
#include <random>

template <typename ValueType = float>
inline void InitData(ValueType* data, int size) {
  std::random_device rnd;
  std::mt19937 mt(rnd());
  std::uniform_int_distribution<> urnd(0, 999);

  for (int i = 0; i < size; ++i) {
    data[i] = (ValueType)(urnd(mt) & 0xff) / 10.0f;
  }

}

template <typename ValueType, int X>
__global__
inline void InitData(ValueType* v1, ValueType* v2) {
  typedef ValueType value_type;
  value_type i = blockIdx.x * blockDim.x + threadIdx.x;
  value_type j = blockIdx.y * blockDim.y + threadIdx.y;

  v1[j * X + i] = j;
  v2[j * X + i] = i;
}

#endif
