#include <climits>
#include <iostream>

template <size_t X, size_t Y>
struct high_bit_helper {
  static const int value =
    ((X >> Y) % 2) ? Y : high_bit_helper<X, Y-1>::value;
};

template <size_t Y>
struct high_bit_helper<Y, 0> {
  static const int value = (Y % 2) ? 0 : 1;
};

template <typename T, T X>
struct static_highest_bit : high_bit_helper<X, CHAR_BIT*sizeof(T)-1> {};

template <typename T, T N>
T doit() {
  std::cout << static_highest_bit<T, N>::value << '\n';
}

auto main() -> int
{
  doit<size_t, 10>();
  return 0;
}
