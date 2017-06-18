#include <iostream>
/**
template <typename T, bool IS_SMALL = (sizeof(T) < sizeof(void*))>
class A;

template <typename T>
class A<T, true> {};

template <typename T>
class A<T, false> {};

A<char> a1;
A<char, true> a2;
**/
template <size_t N>
struct fibonacci {
  static const size_t value = fibonacci<N-1>::value + fibonacci<N-2>::value;
};

template <>
struct fibonacci<0> {
  static const size_t value = 0;
};

template <>
struct fibonacci<1> {
  static const size_t value = 1;
};
/* Another fibonacci
template <size_t N, bool IS_TINY = (N<2)>
struct fibonacci {
  static const size_t value = fibonacci<N-1>::value + fibonacci<N-2>::value;
};

template <size_t N>
struct fibonacci<N, true> {
  static const size_t value = N;
};

template <size_t N, bool IS_TINY>
struct fibonacci_helper {
  static const size_t value =
    fibonacci_helper<N-1>::value + fibonacci_helper<N-2>::value;
};

template <size_t N>
class fibonacci : fibonacci_helper<N, (N<2)> {};
*/
template <typename T, T N>
T doit() {
  std::cout << fibonacci<N>::value << '\n';
}

auto main() -> int
{
  doit<size_t, 10>();
  return 0;
}
