#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <boost/thread.hpp>
#include <CUDA/HPP/CalElapsedTime.hpp>
#include <CUDA/HPP/InitData.hpp>
#include <CUDA/HPP/Transpose.hpp>

typedef float value_type;
int X = std::pow(2, 12);
int Y = std::pow(2, 10);
int x = 16;
int y = 16;
std::size_t NBytes = X * Y * sizeof(int);
dim3 Thread(x, y, 1);
dim3 Block(X / x, Y / y, 1);
int* in;
int* out;

void SpawnSimple() {
  Transpose<int, 4096, 2048><<<Block, Thread>>>(in, out);
}

void SpawnShared() {
  TransposeShared<int, 4096, 2048, 16, 16><<<Block, Thread>>>(in, out);
}

void DoEvaluation() {
  cudaMalloc((void**)&in, NBytes);
  cudaMalloc((void**)&out, NBytes);
  InitData<int, 4096><<<Block, Thread>>>(in, out);
  std::cout << CalElapsedTime<>::Execution(SpawnSimple) << '\n';

  boost::this_thread::sleep_for(boost::chrono::seconds(3));

  InitData<int, 4096><<<Block, Thread>>>(in, out);
  std::cout << CalElapsedTime<>::Execution(SpawnShared) << '\n';

  cudaFree(in);
  cudaFree(out);
}

auto main() -> decltype(0) {
  DoEvaluation();  
  return 0;
}
