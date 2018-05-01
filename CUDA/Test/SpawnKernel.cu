#include "../HPP/SpawnKernel.hpp"

auto main() -> decltype(0) {
  cudaSetupDevice();
  int thread_num = 1 << 24;
  SpawnKernel<>()(thread_num);
  thread_num = 1 << 12;
  SpawnKernel<>()(thread_num);
  cudaDeviceReset();
  return 0;
}
