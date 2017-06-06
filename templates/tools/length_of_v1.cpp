#include <iostream>

template <typename T, size_t N>
size_t length_of(T (&)[N]) {
  return N;
}

// #define length_of(a) sizeof(a)/sizeof(a[0]) // Same

template <typename T>
T doit() {
  T a[10] = {};
  std::cout << length_of(a) << '\n';
}

auto main() -> int
{
  doit<int>();
  return 0;
}
