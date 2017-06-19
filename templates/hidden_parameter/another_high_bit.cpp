#include <climits>
#include <iostream>

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
    U ? (N/2)+helper<U, N-N/2>::value : helper<X, X/2>::value;
};

template <typename T, T x>
T doit() {
//  std::cout << high_bit<T>(x) << '\n';
}

auto main() -> int
{
  doit<size_t, 5>();
  return 0;
}
