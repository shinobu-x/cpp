#include <iostream>

template <typename T>
struct largest_precision_type; /// Nothing!

template < >
struct largest_precision_type<float> {
  typedef float type;
};

template < >
struct largest_precision_type<double> {
  typedef double type;
};

template < >
struct largest_precision_type<int> {
  typedef long type;
};

template <unsigned int N>
struct powerof_2 {
  static const unsigned int value = (1<<N);
};

template <unsigned int N>
struct more_powerof_2 {
  enum { value = (1<<N) };
};

template <typename T>
T doit() {
  auto a = powerof_2<5>::value;
  auto b = more_powerof_2<10>::value;
  largest_precision_type<int>::type c = a + b;
  std::cout << a << '\n';
  std::cout << b << '\n';
  std::cout << c << '\n';
}

auto main() -> int
{
  doit<void>();
  return 0;
}
