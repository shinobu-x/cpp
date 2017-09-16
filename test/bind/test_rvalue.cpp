#include <boost/bind.hpp>

#include <cassert>

int f(int i) {
  return i;
}

auto main() -> decltype(0) {
  assert(
    boost::bind(f, _1)(1) == 1);
  assert(
    boost::bind(f, _2)(1, 2) == 2);
  assert(
    boost::bind(f, _3)(1, 2, 3) == 3);
  assert(
    boost::bind(f, _4)(1, 2, 3, 4) == 4);
  assert(
    boost::bind(f, _5)(1, 2, 3, 4, 5) == 5);
  assert(
    boost::bind(f, _6)(1, 2, 3, 4, 5, 6) == 6);
  assert(
    boost::bind(f, _7)(1, 2, 3, 4, 5, 6, 7) == 7);
  assert(
    boost::bind(f, _8)(1, 2, 3, 4, 5, 6, 7, 8) == 8);
  assert(
    boost::bind(f, _9)(1, 2, 3, 4, 5, 6, 7, 8, 9) == 9);

  return 0;
}
