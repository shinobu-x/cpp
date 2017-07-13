#include <algorithm>
#include <future>
#include <iostream>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

template <typename T>
T do_some(T a, T b) {
  return a*b;
}

template <typename T>
void doit() {

  unsigned long const cpus = std::thread::hardware_concurrency();
  std::vector<std::future<T> > fs(cpus-1);
  std::vector<std::thread> ts(cpus-1);

  for (unsigned long i=0; i<(cpus-1); ++i) {
    std::packaged_task<T(T, T)> t(do_some<T>);
    fs[i] = t.get_future();
    ts[i] = std::thread(std::move(t), 3, 3);
    ts[i].detach();
    std::cout << fs[i].get() << '\n';
  }

}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
