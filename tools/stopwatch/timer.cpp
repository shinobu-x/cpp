#include <chrono>
#include <iostream>
#include <thread>

class timer {
public:
  timer() {
    reset_();
  }

  friend std::ostream& operator<< (std::ostream& out, timer const &t) {
    return out << t.elapsed().count() << "ms";
  }

private:
  std::chrono::high_resolution_clock::time_point s_;

  void reset_() {
    s_ = std::chrono::high_resolution_clock::now();
  }

  std::chrono::milliseconds elapsed() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - s_);
  }
};

auto main() -> decltype(0)
{
  timer t;
  std::this_thread::sleep_for(std::chrono::seconds(3));
  std::cout << t << '\n';
  return 0;
}
