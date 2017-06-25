#include <iostream>

typedef int I;

template <I D1=0, I D2=0, I D3=0, I D4=0, I D5=0>
struct group { /* Empty definition */ }; 

template <typename T>
struct join; /* Empty */

template <I D1, I D2, I D3, I D4, I D5>
struct join<group<D1, D2, D3, D4, D5> > {
  typedef join<group<D1, D2, D3, D4, D5> > next_t;
  const static I pwr10 = 10*next_t::pwr10;
  const static I value = next_t::value+D1*pwr10;
};

template <>
struct join<group<> > {
  const static I pwr10 = 1;
  const static I value = D1;
};

template <typename T>
T doit() {
  std::cout << join<group<1,2,3> >::value << '\n';
}

auto main() -> int
{
  doit<int>();
  return 0;
}
