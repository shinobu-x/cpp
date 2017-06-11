#include <chrono>

#include "stopwatch.hpp"

class timer_base {
public:
  timer_base() : start_(std::chrono::system_clock::time_point::min()){}

  void clear() {
    start_ = std::chrono::system_clock::time_point::min();
  }

  bool is_started() const {
    return (start_.time_since_epoch() != 
      std::chrono::system_clock::duration(0));
  }

  void start() {
    start_ = std::chrono::system_clock::now();
  }

  unsigned long get_ms() {
    if (is_started()) {
      std::chrono::system_clock::duration diff;
      diff = std::chrono::system_clock::now() - start_;
      return (unsigned)(
        std::chrono::duration_cast<std::chrono::milliseconds>(diff).count());
    }
    return 0;
  }
private:
  std::chrono::system_clock::time_point start_;
};

template <typename T, T N>
T doit() {
  timer_base tb;
}

auto main() -> int
{
  doit<int, 100>();
  return 0;
}
