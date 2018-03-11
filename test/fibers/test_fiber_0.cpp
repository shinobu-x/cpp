// Required library
// boost_fiber, boost_context

#include <boost/fiber/all.hpp>
#include <boost/exception_ptr.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

int global_value = 9;

struct exception : public std::runtime_error {
  exception() : std::runtime_error("Oops") {}
};

struct data {
  data() = default;
  data(data const&) = delete;
  data(data&&) = default;
  data& operator=(data const&) = delete;
  data& operator=(data&&) = default;
  int value;
};

void f1(boost::fibers::promise<int>* p, int v) {
  boost::this_fiber::yield();
  p->set_value(v);
}

void f2() {
  boost::fibers::promise<int> p;
  boost::fibers::future<int> f(p.get_future());
  boost::this_fiber::yield();
  boost::fibers::fiber(boost::fibers::launch::post, f1, &p, 1).detach();
  boost::this_fiber::yield();
  assert(1 == f.get());
}

int f3() {
  return 2;
}

void f4() {}

int f5() {
  boost::throw_exception(exception());
  return 3;
}

void f6() {
  boost::throw_exception(exception());
}

int& f7() {
  return global_value;
}

int f8(int v) {
  return v;
}

data f9() {
  data d;
  d.value = 4;
  return d;
}

data f1() {
  boost::throw_exception(exception());
  return data();
}

void create_promise() {
  boost::fibers::promise<int> p1;
  std::allocator<boost::fibers::promise<int> > alloc;
  boost::fibers::promise<int> p2(std::allocator_arg, alloc);
}

auto main() -> decltype(0) {
  return 0;
}
