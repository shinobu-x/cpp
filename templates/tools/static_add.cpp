#include <iostream>

template <size_t N>
struct static_add {
  static const size_t value = N + static_add<N - 1>::value;
};

template <>
struct static_add<0> {
  static const size_t value = 0;
};

template <typename T, T N>
T doit() {
  std::cout <<  static_add<10>::value << '\n';
}

auto main() -> int
{
  doit<int, 10>();
  return 0;
}
