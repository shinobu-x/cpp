#include <algorithm>
#include <future>
#include <iostream>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

template <typename IT, typename T>
T accumulater(IT first, IT last) {
    return std::accumulate(first, last, T());
}

template <typename IT, typename T>
T do_accumulate(IT first, IT last, T init) {
  unsigned long const length = std::distance(first, last);

  if (!length)
    return init;

  unsigned long const cpus = std::thread::hardware_concurrency();

  std::vector<std::future<T> > fs(cpus - 1);
  std::vector<std::thread> ts(cpus - 1);

  unsigned long const size = length/cpus;
  IT start = first;

  for (unsigned long i=0; i<cpus-1; ++i) {
    std::cout << std::this_thread::get_id() << '\n';
    IT end = start;
    std::advance(end, size);
    std::packaged_task<T(IT, IT)> task(accumulater<IT, T>);
    fs[i] = task.get_future();
    ts[i] = std::thread(std::move(task), start, end);
    start = end;
  }

  T r1 = accumulater<IT, T>(start, last);

  std::for_each(ts.begin(), ts.end(), std::mem_fn(&std::thread::join));

  T r2 = init;

  for (unsigned long i=0; i<cpus-1; ++i)
    r1 += fs[i].get();

  r2 += r1;

  return r2;
}

template <typename T>
void doit() {

  std::vector<T> v;
  T init;

 for (T i=0; i<100; ++i)
    v.push_back(i);

 std::cout << do_accumulate(v.begin(), v.end(), init) << '\n';

}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
