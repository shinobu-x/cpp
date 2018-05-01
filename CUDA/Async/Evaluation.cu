// nvcc -std=c++14 CUDA/Async/Evaluation.cu \
//   -lboost_system -lboost_thread -L/src/boost/lib
#include <boost/thread/future.hpp>
#include <boost/thread.hpp>
#include <vector>

#include "../HPP/SpawnKernel.hpp"

auto main() -> decltype(0) {
  cudaSetupDevice();

  std::vector<boost::thread> threads;

  SpawnKernel<> sk;
  int thread_num = 1 << 24;

  for (int i = 0; i < 10; ++i) {
    threads.push_back(boost::thread(boost::ref(sk), thread_num));
  }

  auto it = threads.begin();

  for (; it != threads.end(); ++it) {
    it->join();
  }

  cudaDeviceReset();

  return 0;
}
