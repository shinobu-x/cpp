#include <functional>
#include <iostream>

template <typename T>
struct Do {
  Do(T x) : x_(x) {}
  T x_;
  T& a() { return x_; }
};

template <typename T>
T doit() {
  T a = 10;
  Do<T> b(a);
  auto c = std::mem_fn(&Do<T>::a);
}

auto main() -> int
{
  doit<int>();
  return 0;
}
