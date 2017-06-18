#include <algorithm>
#include <functional>
#include <numeric>
#include <iterator>
#include <thread>
#include <vector>

template <typename I, typename T>
struct accumulate_t {
  void operator() (I first, I last, T& result) {
    result = std::accumulate(first, last, result);
  }
};

template <typename I, typename T>
T accumulater(I first, I last, T init) {
  unsigned long const length = std::distance(first, last);

  if (!length)
    return init;

  unsigned long const min_per_thread = 25;
  unsigned long const max_threads = (length + min_per_thread-1)/min_per_thread;

  unsigned long const hardware_threads = std::thread::hardware_concurrency();

  unsigned long const num_threads =
    std::min(hardware_threads != 0 ? hardware_threads : 2, max_threads);

  unsigned long const size = length / num_threads;

  std::vector<T> results(num_threads);
  std::vector<std::thread> threads(num_threads - 1);

  I start = first;

  for (unsigned long i=0; i<(num_threads-1); ++i) {
    I end = start;
    std::advance(end, size);
    threads[i] = std::thread(
      accumulate_t<I, T>(), start, end, std::ref(results[i]));
    start = end;
  }

  accumulate_t<I, T>()(start, last, results[num_threads-1]);

  std::for_each(threads.begin(), threads.end(),
    std::mem_fn(&std::thread::join));

  return std::accumulate(results.begin(), results.end(), init);
}

template <typename T>
T doit() {
  typedef typename std::vector<T>::iterator iterator_t;
  std::vector<T> v{1, 2, 3, 4, 5};
  T init;
  iterator_t start = v.begin();
  iterator_t last = v.end();
  accumulater<iterator_t, T>(start, last, init);
}

auto main() -> int
{
  doit<int>();
  return 0;
}
