#include <boost/bind.hpp>
#include <boost/ref.hpp>

#include <cassert>

struct X {
  int f(int x) {
    return x;
  }

  int g(int x) const {
    return x*x;
  }
};

auto main() -> decltype(0) {
  X o;
  assert(
    boost::bind(&X::f, _1, 1)(boost::ref(o)) == 1);

  assert(
    boost::bind(&X::g, _1, 2)(boost::cref(o)) == 2*2);

  return 0;
}
