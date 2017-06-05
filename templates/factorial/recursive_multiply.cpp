#include <iostream>

template <int N>
struct recursive_multiply {
  enum { value = recursive_multiply<N - 1>::value * N };
};

template < >
struct recursive_multiply<0> {
  enum { value = 1 }; /// Stop!
};

template <typename T, T N>
T doit() {
  const T v = recursive_multiply<N>::value;
  std::cout << v << '\n';
}

auto main() -> int
{
  doit<int, 5>();
}
