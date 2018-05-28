#include <iostream>
#include <utility>
#include <boost/timer/timer.hpp>

#define N 20

template <typename T>
T Fibonacci1(T n) {
  if (n == 0 || n == 1) {
    return n;
   } else {
    return Fibonacci1(n - 1) + Fibonacci1(n - 2);
   }
}

template <typename T>
T Fibonacci2(T n) {
  if (n == 0) {
    return n;
  }

  std::pair<int, int> v{0, 1};

  for (T i = 1; i < n; ++i) {
    v = {v.second, v.first + v.second};
  }
  return v.second;
}

void DoFib1() {
  boost::timer::cpu_timer t;
  Fibonacci1(N);
  std::cout << t.format() << "\n";
}

void DoFib2() {
  boost::timer::cpu_timer t;
  Fibonacci2(N);
  std::cout << t.format() << "\n";
}

auto main() -> decltype(0) {
  DoFib1();
  DoFib2();
  return 0;
}
