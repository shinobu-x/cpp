#include <iostream>

template <typename T>
struct add_ref {
  typedef T& type;
  T a;
};

template <typename T>
struct add_ref<T&> {
  typedef T& type;
  T a;
};

template <>
struct add_ref<void> {};

template <typename T>
T dothat(T& x) {
  x += x*x;
}

template <typename T>
T doit() {
  add_ref<T> ref;
  ref.a = 5;
  dothat<T>(ref.a);
  std::cout << ref.a << '\n';
}

auto main() -> int
{
  doit<int>();
  return 0;
}
