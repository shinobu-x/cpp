#include <climits>
#include <iostream>

template <typename T, T VALUE>
struct static_parameter {};

template <typename T, T VALUE>
struct static_value : static_parameter<T, VALUE> {
  const static T value = VALUE;
};

template <size_t X, size_t Y>
struct helper {
  static const size_t v = (X >> (Y/2));
  static const int value = 
    (v ? Y/2 : 0) + helper<(v ? v : X), (v ? Y-Y/2 : Y/2)>::value;
};

template <size_t X>
struct helper<X, 1> {
  static const int value = X ? 0 : -1;
};

template <size_t X>
struct static_highest_bit
  : static_value<int, helper<X, CHAR_BIT*sizeof(size_t)>::value> {};

template <typename T, T V>
T doit() {
  static_highest_bit<V> bit;
  std::cout << bit.value << '\n';
}

auto main() -> int
{
  doit<size_t, 10>();
  return 0;
}
