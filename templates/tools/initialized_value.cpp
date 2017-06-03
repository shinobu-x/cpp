#include <iostream>

template <typename T>
struct initialized_value {
  T result;

  initialized_value() : result() {}
};

template <typename T>
T doit() {
  initialized_value<T> a;
  initialized_value<T [5]> b;
  initialized_value<std::string> c;

  std::cout << a.result << '\n';
  for (auto v : b.result)
    std::cout << v << '\n';
}

auto main() -> int
{
  doit<double>();
  return 0;
}
