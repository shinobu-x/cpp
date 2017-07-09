#include <iostream>

template <typename T>
constexpr T sq_helper(T sq, T d, T a) {
  return sq <= a ? sq_helper(sq+d,d+2,a) : d;
}

template <typename T>
constexpr T sq(T x) {
  return sq_helper<T>(1,3,x)/2 - 1;
}

template <typename T>
void doit() {
  T a = 5;
  std::cout << sq<T>(a) << '\n';
}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
