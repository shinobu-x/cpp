#include <iostream>
#include <thread>

class thread_guard {
public:
  thread_guard(std::thread& t) : t_(t) {};

  ~thread_guard() {
    if (t_.joinable())
      t_.join();
  }

  thread_guard(thread_guard const&) = delete;
  thread_guard operator= (thread_guard const&) = delete;
private:
  std::thread& t_;
};

template <typename T>
T doit() {
  T i = 1;
  std::thread t(
    [&i](T x){
      while (i != x) {
        std::cout << x << '\n';
        x = (x - i);
      }
    }, 10);

  thread_guard g(t);
}
auto main() -> int
{
  doit<int>();
  return 0;
}
