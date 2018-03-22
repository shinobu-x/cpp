#include <iostream>

void f() {}

template <typename T, typename... Ts>
void f(T a, Ts ...as) {
  std::cout << a << '\n';
  f(as...);
}

void doit() {
  f(1, 2, 3, 4, 5, 6, 7, 8, 9);
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
