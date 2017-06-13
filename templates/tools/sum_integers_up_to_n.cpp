#include <iostream>

template <typename T, T VALUE>
struct static_parameter {};

template <typename T, T VALUE>
struct static_value : static_parameter<T, VALUE> {
  const static T value = VALUE;
};

template <size_t N>
struct sum_integers_up_to_n {
  static const size_t value = N + sum_integers_up_to_n<N - 1>::value;
};

template <>
struct sum_integers_up_to_n<0> {
  static const size_t value = 0;
};

template <typename T, T N>
T doit() {
  std::cout <<  sum_integers_up_to_n<10>::value << '\n';
}

auto main() -> int
{
  doit<int, 10>();
  return 0;
}
