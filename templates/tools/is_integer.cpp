#include <cstddef>
#include <iostream>

template <typename T>
struct is_integer {
  const static bool value = false;
};

// Specialization here

template <>
struct is_integer<short> {
  const static bool value = true;
};

template <>
struct is_integer<int> {
  const static bool value = true;
};

// template <>
// struct is_integer<long> {
//   const static bool value = true;
// };

template <>
struct is_integer<ptrdiff_t> {
  const static bool value = true;
};

auto main() -> int
{
  return 0;
}
