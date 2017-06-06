#include <cmath>
#include <iostream>

struct naive_algorithm_tag {};
struct precise_algorithm_tag {};

template <typename T>
inline T log1p(T x, naive_algorithm_tag) {
  return log(x + 1);
}

template <typename T>
inline T log1p(T x, precise_algorithm_tag) {
  const T xp1 = x + 1;
  return xp1 == 1 ? x : x*std::log(xp1)/(xp1 - 1);
}

template <typename T>
T doit() {
  T v1 = log1p<T>(3.14, naive_algorithm_tag());
  T v2 = log1p(0.0000000000314, precise_algorithm_tag());
  std::cout << v1 << '\n';
  std::cout << v2 << '\n';
}

auto main() -> int
{
  doit<double>();
  return 0;
}
