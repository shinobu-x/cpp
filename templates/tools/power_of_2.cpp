#include <iostream>

template <int I>
struct power_of_2 {
  static const int value = 2 * power_of_2<I - 1>::value;
};

template <>
struct power_of_2<0> {
  static const int value = 1;
};

template <typename T, T N>
T doit() {
  std::cout << power_of_2<N>::value << '\n';
}

auto main() -> int
{
  doit<int, 5>();
  return 0;
}
