#include <iostream>

template <typename T>
T caller(T x) { return 1 + x; }

template <typename R, typename A>
inline R apply(R (*F)(A), A arg) {
  return F(arg);
}

template <typename T>
T doit() {
  T x = apply(&caller<T>, 1.23);
  return x;
}

auto main() -> int
{
  std::cout << doit<double>() << '\n';
  return 0;
}
