#include <iostream>

template <typename OUTER_T>
class Outer {
public:
  template <typename INNER_T>
  OUTER_T f(OUTER_T x, INNER_T y) const {
    return x*y;
  }
};

template <typename T, typename U>
void doit(T x, U y) {
  T z = Outer<T>().f(x, y);

  std::cout << z << '\n';
}

auto main() -> int
{
  doit(3.14, 3);

  return 0;
}
