#include <iostream>

template <typename... A>
struct pack {
  int value;
};

template <typename T>
T doit() {
  pack<T> a;
  pack<T, double> b;
  pack<T, double, float> c;
}

auto main() -> int
{
  doit<int>();
  return 0;
}
