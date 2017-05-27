#include <iostream>

/**
 * We need to write a specialization when there are some overloaded templates
 */
template <typename T>
void f(const T& x) { std::cout << x << '\n'; }

template <typename T>
void f(T* x) { std::cout << x << '\n'; }

template <typename T>
void doit() {
  T a = 1;
  const T& b = 2;
  const T *c = &b;

  f(a);
  f(b);
  f(*c);
  
}

auto main() -> int
{
  doit<int>();
  return 0;
}
