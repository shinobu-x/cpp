#include <iostream>

template <size_t M, size_t N>
struct static_raise {
  const static size_t value = M * static_raise<M, N - 1>::value;
};

template <size_t M>
struct static_raise<M, 0> {
  const static size_t value = 1;
};

template <typename T, T M, T N>
T doit() {
  std::cout << static_raise<M, N>::value << '\n';
}

auto main() -> int
{
  doit<size_t, 5, 10>();
  return 0;
} 
