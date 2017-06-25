#include <iostream>

typedef int I;

template <I D1, I D2=0, I D3=0, I D4=0, I D5=0>
struct join {
  typedef join<D2, D3, D4, D5> next_t;
  static const I pwr10 = 10*next_t::pwr10;
  static const I value = next_t::value+D1*pwr10;
};

template <I D1>
struct join<D1> {
  static const I pwr10 = 1;
  static const I value = D1;
};

template <typename T>
T doit() {
  std::cout << join<1,2,3>::value << '\n';
}

auto main() -> int
{
  doit<int>();
  return 0;
}
