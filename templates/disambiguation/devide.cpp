#include <iostream>

template <int N, typename T>
struct A {
  int get_N() {
    return N;
  }
};

template <int N>
struct B : A<N%2, B<N> >, B<N/2> {
  int get_N() {
    return A<N%2, B<N> >::get_N();
  }
};

template <>
struct B<0> {};

template <typename T, T N>
T doit() {
  B<N> b;
  std::cout << b.get_N() << '\n';
}

auto main() -> int
{
  doit<int, 10>();
  return 0;
}
