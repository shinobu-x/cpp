#include <iostream>
#include <thread>
#include <utility>

struct Callable {
  void operator()() {
    std::cout << __func__ << '\n';
  }
};

template <typename F>
struct Task {
  F f_;
  F callable() {
    return std::move(f_);
  }
};

void doit() {
  Task<Callable> t;
  auto f = t.callable();
  f();
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
