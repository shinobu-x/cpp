#include <iostream>
#include <type_traits>

struct A {
  operator int () const { return value_; }
private:
  int value_;
};

struct B {
  operator A () { return a; }
private:
  A a;
};

auto main() -> decltype(0)
{
  B b;
  int a = A(b);
  std::cout << a << '\n';
  return 0;
}
