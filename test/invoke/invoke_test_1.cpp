#include <boost/thread/detail/invoke.hpp>

#include <cassert>

int f() {
  return 1;
}

struct invoke_test_1 {

  int operator()() {
    return 2;
  }

  int operator()(int a) {
    return a;
  }

  int operator()() const;

};

int invoke_test_1::operator()() const {
  return 4;
}

void do_test_invoke_1() {
  invoke_test_1 obj;
  const invoke_test_1 cobj;

  assert(f() == 1);
  assert(invoke_test_1()() == 2);
  assert(invoke_test_1()(3) == 3);
  assert(boost::detail::invoke(f) == 1);
  assert(boost::detail::invoke(&f) == 1);
  assert(boost::detail::invoke<int>(f) == 1);
  assert(boost::detail::invoke<int>(&f) == 1);
  assert(boost::detail::invoke(invoke_test_1()) == 2);
  assert(boost::detail::invoke<int>(invoke_test_1()) == 2);
  assert(boost::detail::invoke(invoke_test_1(), 3) == 3);
  assert(boost::detail::invoke<int>(invoke_test_1(), 3) == 3);
  assert(boost::detail::invoke(obj) == 2);
  assert(boost::detail::invoke<int>(obj) == 2);
  assert(boost::detail::invoke(obj, 3) == 3);
  assert(boost::detail::invoke<int>(obj, 3) == 3);
  assert(boost::detail::invoke(cobj) == 4);
  assert(boost::detail::invoke<int>(cobj) == 4);

}

auto main() -> decltype(0) {
  do_test_invoke_1();
  return 0;
}
