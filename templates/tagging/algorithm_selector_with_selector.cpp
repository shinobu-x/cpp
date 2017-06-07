#include <cmath>
#include <iostream>

template <bool condition>
struct selector {};

template <typename T>
inline T log1p(T x, selector<true>) {
  return log(x + 1);
}

template <typename T>
inline T log1p(T x, selector<false>) {
  const T xp1 = x + 1;
  return xp1 == 1 ? x : x*std::log(xp1)/(xp1 - 1);
}

template <typename T>
T doit() {
  T v1 = log1p<T>(3.14, selector<true>());
  T v2 = log1p(0.0000000000314, selector<false>());
  std::cout << v1 << '\n';
  std::cout << v2 << '\n';
}

auto main() -> int
{
  doit<double>();
  return 0;
}
