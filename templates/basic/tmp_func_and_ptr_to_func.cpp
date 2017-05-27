#include <iostream>

template <double F()>
struct A {
  operator double() const {
    return F();
  }
};

template <double (*F)()>
struct B {
  operator double() const {
    return F();
  }
};

double f() {
  return 3.14;
}

auto main() -> int
{
  A<f> a;
  B<f> b;

  std::cout << a << '\n';
  std::cout << b << '\n';

  return 0;
}
