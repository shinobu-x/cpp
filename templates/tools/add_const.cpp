#include <iostream>

template <bool condition>
struct selector {};

template <typename T>
struct instance_of {
  instance_of(int = 0) {}
  ~instance_of() {}
  typedef T type;
};

template <typename T>
struct is_const : selector<true> {};

template <typename T>
struct add_const /*: instance_of<const T>*/ {
  typedef typename T::type const type;
};

template <typename T>
struct add_const<const T> /*: instance_of<const T>*/ {
  typedef typename T::type type;
};

template <typename T>
T doit() {
  typedef typename add_const<instance_of<T> >::type type_t;
  type_t a = 1;
}

auto main() -> int
{
  doit<int>();
  return 0;
}
