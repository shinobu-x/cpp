#include <iostream>

template <int I>
struct power_of_10 {
  static const int value = 10 * power_of_10<I - 1>::value;
};

template <>
struct power_of_10<0> {
  static const int value = 1;
};

template <int I, int N>
struct digit {
  static const int value = (N / power_of_10<I>::value) % 10;
};

template <typename T>
T doit() {
  const T a = 2, b = 12345;
  std::cout << digit<a, b>::value << '\n';
}

auto main() -> int
{
  doit<int>();
  return 0;
}
