#include <iostream>

typedef int I;

template <I D1, I D2 = 0, I D3 = 0, I D4 = 0, I D5 = 0>
struct join {
  static const I value = join<D2, D3, D4, D5>::value * 10 + D1;
};

template <I D1>
struct join<D1> {
  static const I value = D1;
};

template <typename T>
T doit() {
  std::cout << join<3,2,1>::value << '\n';
}

auto main() -> int
{
  doit<int>();
  return 0;
}
