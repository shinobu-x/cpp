#include <boost/bind.hpp>

#include <boost/function.hpp>
#include <iostream>

template <typename F>
void doit(F&& f) {
  boost::function0<void> fn;
  fn = boost::bind(f, 1);
  fn();
}

void f(int i) {
  std::cout << i << '\n';
}

auto main() -> decltype(0) {
  doit(f);
  return 0;
}
