#include <iostream>

template <typename A>
A DoThis(A a) {
  if (a) {}
}

template <typename B>
B DoThat(B a, B b) {
  if (DoThis(a + b)) {}
}

auto main() -> decltype(0) {
  if (std::cout << DoThat(1, 5) << "\n") {}
}
