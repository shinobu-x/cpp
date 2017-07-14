#include <chrono>
#include <iostream>
#include <future>
#include <thread>

auto f(int a) -> decltype(a) {
  return a;
}

template <typename T>
void doit() {
  {
    T x = 3;
    std::future<T> a = std::async(std::launch::async, f, x);
    T r = a.get();
    std::cout << std::this_thread::get_id() << '\n';
    std::this_thread::sleep_for(std::chrono::seconds(3));
    std::cout << r << '\n';
  }

  {
    T x = 2;
    std::future<T> a = std::async(std::launch::deferred, f, x);
    T r = a.get();
    std::cout << std::this_thread::get_id() << '\n';
    std::this_thread::sleep_for(std::chrono::seconds(3));
    std::cout << r << '\n';
  }
}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
