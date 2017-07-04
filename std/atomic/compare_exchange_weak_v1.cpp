#include <atomic>
#include <thread>
#include <iostream>

/**
 * bool compare_exchange_weak(T& expected, T desired,
 *   memory_order success, memory_order failure) volatile noexcept;
 * bool compare_exchange_weak(T& expected, T desired,
 *   memory_order success, memory_order failure) noexcept;
 *
 * bool compare_exchange_weak(T& expected, T desired,
 *   memory_order order = memory_order_seq_cst) volatile noexcept;
 * bool compare_exchange_weak(T& expected, T desired,
 *   memory_order order = memory_order_seq_cst) noexcept;
 */

template <typename T>
T doit() {
  std::atomic<T> a(3);
  T expected = 3;
  bool r = a.compare_exchange_weak(expected, 2);
  std::cout << r << " " << a.load() << " " << expected << '\n';

  std::thread t([&a]{
    T expected = 1;
    bool r = a.compare_exchange_weak(expected, 2);
    std::cout << r << " " << a.load() << " " << expected << '\n';
  });

  t.join();
}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
