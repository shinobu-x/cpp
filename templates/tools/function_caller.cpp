#include <iostream>

template <typename T>
T f(T a) {
  return a;
}

template <typename T>
struct instance_of {
  instance_of(int=0) {}
  ~instance_of() {}

  typedef T type;
};

template <typename A, typename R>
R do_f(R (*f)(A), typename instance_of<R>::type x) {
  return f(x);
}

/* We want to template this
int do_ff(int (*f)(int), int x) {
  return f(x);
} */

template <typename T>
T doit() {
//  std::cout << do_ff(f<int>, 5) << '\n';
  std::cout << do_f<T>(f<T>, 5) << '\n';
}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
