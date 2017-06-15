#include <iostream>

template <int I, int N>
struct digit;

template <int I, int N>
struct digit {
  static const int value = digit<I-1, N/10>::value;
};

template <int N>
struct digit<0, N> {
  static const int value = (N % 10);
};

template <typename T>
T doit() {
  const T a = 2, b = 33893;
  std::cout << digit<a, b>::value << '\n';
}

auto main() -> int
{
  doit<int>();
  return 0;
}
