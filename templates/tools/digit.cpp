#include <iostream>

template <int I, int N>
struct digit;

template <int I, int N>
struct digit {
  static const int value = digit<i-1, N/10>::value;
};

template <int N>
struct digit<0, N> {
  static const int value = (N % 10);
};

template <typename T>
T doit() {
}

auto main() -> int
{
  return 0;
}
