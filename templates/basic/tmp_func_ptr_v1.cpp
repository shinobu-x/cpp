#include <iostream>

double f(double x) {
  return x + 1;
}

template <typename T>
T g(T x) {
  return x + 1;
}

typedef double (*FUNC_T)(double);

template <typename T>
void doit() {
  T a = 3.14;
  FUNC_T f1 = f;
  FUNC_T f2 = g<T>;

  std::cout << f1(a) << '\n';
  std::cout << f2(a) << '\n';
}

auto main() -> int
{
  doit<double>();
  return 0;
}
