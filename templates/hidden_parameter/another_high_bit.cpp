#include <climits>
#include <iostream>

template <typename T, T V>
struct static_parameter {};

template <typename T, T V>
struct static_value : static_parameter<T, V> {
  const static T value = V;
};

template <typename T>
T high_bit(T x, T y = CHAR_BIT*sizeof(T)) {
  T u = (x>>(y/2));
  
  if (u>0) return high_bit(u, y-y/2) + (y/2);
  else return high_bit(x, y/2);
}

template <size_t X, int N>
struct helper {
  static const size_t U = (X>>(N/2));
  static const int value = 
    U ? (N/2) : 0 + helper<(U ? U : X), (U ? N-N/2 : N/2)>::value;
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
  std::cout << static_highest_bit<V>::value << '\n';
}

auto main() -> int
{
  doit<size_t, 5>();
  return 0;
}
