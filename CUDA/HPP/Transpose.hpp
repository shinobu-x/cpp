#include <cuda_runtime.h>

template <typename ValueType, int X, int Y>
__global__
void Transpose(ValueType* in, ValueType* out) {
  typedef ValueType value_type;
  value_type i = blockIdx.x * blockDim.x + threadIdx.x;
  value_type j = blockIdx.y * blockDim.y + threadIdx.y;

  out[i * Y + j] = in[j * X + i];
}

template <typename ValueType, int X, int Y, int Tx, int Ty>
__global__
void TransposeShared(ValueType* in, ValueType* out) {
  typedef ValueType value_type;
  __shared__ value_type tmp[Tx][Ty + 1];
  value_type i = blockIdx.x * blockDim.x + threadIdx.x;
  value_type j = blockIdx.y * blockDim.y + threadIdx.y;

  tmp[threadIdx.x][threadIdx.y] = in[j * X + i];
  __syncthreads();

  i = blockIdx.y * blockDim.y + threadIdx.x;
  j = blockIdx.x * blockDim.x + threadIdx.y;

  out[j * Y + i] = tmp[threadIdx.y][threadIdx.x];
}
