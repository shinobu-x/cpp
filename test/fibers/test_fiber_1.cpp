#include <boost/fiber/all.hpp>
#include <boost/throw_exception.hpp>
#include <stdexcept>

int v = 1;

struct xp : public std::runtime_error {
  xp() : std::runtime_error("Opps") {}
};

struct data {
  data() = default;
  data(data const&) = delete;
  data(data&&) = default;
  data& operator=(data const&) = delete;
  data& operator=(data&&) = default;
  int value = 0;
};

void f1(boost::fibers::promise<int>* p, int i) {
  boost::this_fiber::yield();
  p->set_value(i);
}

void f2() {
  boost::fibers::promise<int> p;
  boost::fibers::future<int> f = p.get_future();
  assert(f.valid());
  boost::this_fiber::yield();
  boost::fibers::fiber(boost::fibers::launch::dispatch, f1, &p, 2).detach();
  boost::this_fiber::yield();
  assert(2 == f.get());
}

int f3() {
  return 3;
}

void f4() {}

int f5() {
  boost::throw_exception(xp());
  return 5;
}

void f6() {
  boost::throw_exception(xp());
}

int& f7() {
  return v;
}

int f8(int v) {
  return v;
}

data f9() {
  data d;
  d.value = 9;
  return d;
}

data f10() {
  data d;
  boost::throw_exception(xp());
  return d;
}

void test_create() {
  boost::fibers::promise<int> p1;
  std::allocator<boost::fibers::promise<int> > alloc;
  boost::fibers::promise<int> p2(std::allocator_arg, alloc);
}

void test_create_ref() {
  boost::fibers::promise<int&> p1;
  std::allocator<boost::fibers::promise<int&> > alloc;
  boost::fibers::promise<int&> p2(std::allocator_arg, alloc);
}

void test_create_void() {
  boost::fibers::promise<void> p1;
  std::allocator<boost::fibers::promise<void> > alloc;
  boost::fibers::promise<void> p2(std::allocator_arg, alloc);
}

void doit() {
  test_create();
  test_create_ref();
  test_create_void();
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
