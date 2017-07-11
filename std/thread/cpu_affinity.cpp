#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include <sched.h>  /** sched_getcpu **/

// taskset -c 1,2 ./a.out

template <typename T>
void doit() {
  std::mutex m;
  std::vector<std::thread> vt;
  T cpus = std::thread::hardware_concurrency();

  for (T i=0; i<cpus; ++i)
    vt.push_back(
      // while (true) {
      std::thread(
        [&m, i]{
          std::lock_guard<std::mutex> l(m);
          std::cout << "Thread# " << i
            << " ID:" << std::this_thread::get_id()
            << " CPU: " << sched_getcpu() << '\n';

          std::this_thread::sleep_for(std::chrono::seconds(3));
      // }
  }));

  for (typename std::vector<std::thread>::iterator it=vt.begin();
    it!=vt.end(); ++it)
    it->join();
}

auto main() -> decltype(0)
{
  doit<unsigned>();
  return 0;
}
