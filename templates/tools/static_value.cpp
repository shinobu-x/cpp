#include <iostream>

template <typename T, T VALUE>
struct static_parameter {};

template <typename T, T VALUE>
struct static_value : static_parameter<T, VALUE> {
  const static T value = VALUE;
};

template <typename T, T VALUE>
inline T static_value_cast(static_value<T, VALUE>) {
  return VALUE;
}

template <typename T>
T doit() {
  static_value<T, 3> b;
  std::cout << b.value << '\n';
  // b.value = 5;
  T c = static_value_cast(b);
  std::cout << c << '\n';
  c = 5;
  std::cout << c << '\n';
}

auto main() -> int
{
  doit<int>();
  return 0;
}
