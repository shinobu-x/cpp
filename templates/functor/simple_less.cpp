#include <iostream>

template <typename T>
struct LESS {
  bool operator()(const T& x, const T& y) const {
    return (x < y);
  }
};

template <typename T>
T doit() {
  const T& a = 3, b = 4;

  LESS<T> l;
  std::cout << l(a, b) << '\n';
}

auto main() -> int
{
  doit<int>();

  return 0;
}
