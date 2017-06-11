#include <iostream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

class scoped_thread {
public:
  explicit scoped_thread(std::thread t) : t_(std::move(t)) {
    if (!t_.joinable()) 
      throw std::logic_error("No available thread is running");
  }

  ~scoped_thread() {
    t_.join();
  }

  scoped_thread(scoped_thread const&) = delete;
  scoped_thread& operator= (scoped_thread const&) = delete;
private:
  std::thread t_;
};

template <typename T>
T doit() {
  scoped_thread st(std::thread([]{
    std::cout << std::this_thread::get_id() << '\n';
  }));
}

auto main() -> int
{
  doit<int>();
  return 0;
}
