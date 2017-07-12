#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include <sched.h>    /** cpu_set_t **/
#include <pthread.h>  /** pthread_setaffinity_np **/

template <typename T>
void doit() {
  T cpus = std::thread::hardware_concurrency();
  std::mutex m;
  std::vector<std::thread> vt((int)cpus);

  for (T i=0; i<cpus; ++i) {
    vt[i] = std::thread([&m, i] {
      std::this_thread::sleep_for(std::chrono::seconds(2));

      while (true) {
 
        {
          std::lock_guard<std::mutex> l(m);
          std::cout << "Thread: " << i << '\n'
            << "CPU: " << sched_getcpu() << '\n';
        }

        std::this_thread::sleep_for(std::chrono::seconds(2));
      }

    });

    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(i, &cs);   

    /**
     * @pthread_t thread
     * @size_t cpusetsize
     * @const cpu_set_t* cpuset | cpu_set_t* cpuset
     */
    int r = pthread_setaffinity_np(vt[i].native_handle(),
      sizeof(cpu_set_t), &cs);

    if (r != 0)
      std::cerr << "Error: pthread_setaffinity_np" << '\n';
  }

  for (auto& t : vt)
    t.join();
}

auto main() -> decltype(0)
{
  doit<unsigned>();
  return 0;
}
