#include <numeric>
#include <iostream>
#include <future>
#include <vector>

template <typename IT, typename T>
struct accumulate_t {
  T operator() (IT f, IT l) {
    return std::accumulate(f, l, T());
  }
};

template <typename IT, typename T>
T do_accumulate(IT f, IT l, T init) {
  unsigned long const length = std::distance(f, l);

  if (!length)
    return init;

  T r = accumulate_t<IT, T>()(f, l);
}

template <typename T>
void doit() {
  std::vector<T> v;

  for (T i=0; i<100; ++i)
    v.push_back(i);

  T init = 0;

  std::cout << do_accumulate(v.begin(), v.end(), init) << '\n';
}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
