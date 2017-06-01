#include <algorithm>
#include <iostream>

template <size_t N>
struct is_prime {
  enum { value = 0 };
};

template < >
struct is_prime<2> {
  enum { value = 1 };
};

template < >
struct is_prime<3> {
  enum { value = 1 };
};

template <typename T>
T doit() {
  T data[10];
  std::fill_n(data, is_prime<3>::value, 3.14);
  for (auto v : data) 
    std::cout << v << '\n';
}

auto main() -> int
{
  doit<double>();
  return 0;
}
