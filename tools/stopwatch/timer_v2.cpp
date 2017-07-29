#include <chrono>
#include <iostream>
#include <thread>

class timer {
public:
  timer() : start_(clock_::now()) {}

  void reset() {
    reset_();
  }

  friend std::ostream& operator<< (std::ostream& out, timer const& t) {
    return out << t.elapsed_().count() << "ms";
  }

private:
  typedef std::chrono::high_resolution_clock clock_;
  std::chrono::high_resolution_clock::time_point start_;

  void reset_() {
    start_ = clock_::now();
  }

  std::chrono::milliseconds elapsed_() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
      clock_::now() - start_);
  }
};

auto main() -> decltype(0)
{
  timer t;
  std::this_thread::sleep_for(std::chrono::seconds(3));
  std::cout << t << '\n';
}
