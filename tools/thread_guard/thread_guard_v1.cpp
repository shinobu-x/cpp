#include <iostream>
#include <thread>

class thread_guard {
public:
  explicit thread_guard(std::thread& t) : t_(t) {}

  ~thread_guard() {
    if (!t_.joinable()) {
      throw std::logic_error("!");
    }
    t_.join();
  }

  thread_guard(thread_guard const&) = delete;
  thread_guard& operator= (thread_guard const&) = delete;

private:
  std::thread& t_;
};

auto main() -> decltype(0) {
  std::thread t([]{ std::cout << __func__ << '\n'; });
  thread_guard tg(t);  
  return 0;
}
