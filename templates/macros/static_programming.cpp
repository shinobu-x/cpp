#include <iostream>

template <size_t S>
struct uint_n;

#define M_UINT_N(T, N) template<> struct uint_n<N> { typedef T type; }

template <typename T, T N>
T doit() {
  M_UINT_N(T, N);
}
auto main() -> int
{
  doit<int, 10>();
  return 0;
}
