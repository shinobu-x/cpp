#include <algorithm>
#include <iostream>

template <typename T, T N = 5>
T doit() {
  T a = 3, b = a + N;

  std::cout << a << " " << b << '\n';

  std::swap(a, b);

  std::cout << a << " " << b << '\n';
}

auto main() -> int
{
  doit<int>();
  return 0;
}
