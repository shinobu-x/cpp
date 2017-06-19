#include <future>
#include <iostream>
#include <thread>

typedef size_t type;

template <type V>
struct do_fibo {
  static const type value = do_fibo<V-1>::value + do_fibo<V-2>::value;
};

template <>
struct do_fibo<0> {
  static const type value = 0;
};

template <>
struct do_fibo<1> {
  static const type value = 1;
};

template <typename T>
void do_some(std::promise<int> p, int x) {
  try {
    int r = x*x;
    p.set_value(r);  // Set return to promise
  } catch (...) {
    p.set_exception(std::current_exception());  // Set exception to promise
  }
}

template <typename T, T N>
T doit() {
  std::promise<T> p;
  std::future<T> f = p.get_future();

  T x = N;
  std::thread t(do_some<T>, std::move(p), x);  // Calculate on different thread

  T r = f.get();

  t.join();

  std::cout << r << '\n';

  return 0;
}

auto main() -> int
{
  doit<int, 100>();
  return 0;
}
