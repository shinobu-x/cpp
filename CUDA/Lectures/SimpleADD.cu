#include <cstdlib>
#include <cuda_runtime.h>
#include <random>

template <typename T>
__global__
void Init(T* a, T* b, T* c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  a[i] = 1.0;
  b[i] = 2.0;
  c[i] = 0.0;
}

template <typename T>
__global__
void Add(T* a, T* b, T* c, T A, T B) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  c[i] = a[i]*A + b[i]*B;
}

auto main() -> decltype(0) {
  typedef float value_type;
  int N = 2 << 20;
  std::size_t NBytes = N * sizeof(value_type);
  int T = 256;

  std::random_device rnd;
  std::mt19937 mt(rnd());
  std::uniform_int_distribution<> urnd(0, 999);

  value_type* a;
  value_type* b;
  value_type* c;
  auto A = (value_type)(urnd(mt) & 0xff) / 10.0f;
  auto B = (value_type)(urnd(mt) & 0xff) / 10.0f;

  cudaMalloc(&a, NBytes);
  cudaMalloc(&b, NBytes);
  cudaMalloc(&c, NBytes);

  dim3 Block(T);
  dim3 Grid((T + Block.x - 1) / Block.x);

  Init<<<Grid, Block>>>(a, b, c);
  Add<<<Grid, Block>>>(a, b, c, A, B);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

  return 0;
}
