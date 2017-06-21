#include <iostream>

template <typename T>
struct base {
  typedef T type;
};
/*
template <typename T>
struct derived : base<T> {
  type t;
};
*/
template <typename T>
struct type_of {
  typedef typename T::type type;
};

template <typename T>
T doit() {
  typename type_of<base<T> >::type t;
  t = 0;
  std::cout << t << '\n';
  std::cout << sizeof(t) << '\n';
}

auto main() -> int
{
  doit<int>();
  return 0;
}
