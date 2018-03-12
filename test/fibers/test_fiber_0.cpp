// Required library
// boost_system, boost_context, boost_fiber, boost_filesystem

#include <boost/fiber/all.hpp>
#include <boost/exception_ptr.hpp>

#include <memory>
#include <stdexcept>
#include <string>

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

void create() {
  boost::fibers::promise<int> p1;
  std::allocator<boost::fibers::promise<int> > alloc;
  boost::fibers::promise<int> p2(std::allocator_arg, alloc);
}

void create_ref() {
  boost::fibers::promise<int&> p1;
  std::allocator<boost::fibers::promise<int&> > alloc;
  boost::fibers::promise<int&> p2(std::allocator_arg, alloc);
}

void test_move() {
  boost::fibers::promise<int> p1;
  boost::fibers::promise<int> p2(boost::move(p1));
  p1 = boost::move(p2);
}

void test_move_ref() {
  boost::fibers::promise<int&> p1;
  boost::fibers::promise<int&> p2(boost::move(p1));
  p1 = boost::move(p2);
}

void test_move_void() {
  boost::fibers::promise<void> p1;
  boost::fibers::promise<void> p2(boost::move(p1));
  p1 = boost::move(p2);
}

void test_swap() {
  boost::fibers::promise<int> p1;
  boost::fibers::promise<int> p2(boost::move(p1));
  p1.swap(p2);
}

void test_swap_ref() {
  boost::fibers::promise<int&> p1;
  boost::fibers::promise<int&> p2(boost::move(p1));
  p1.swap(p2);
}

void test_swap_void() {
  boost::fibers::promise<void> p1;
  boost::fibers::promise<void> p2(boost::move(p1));
  p1.swap(p2);
}

void test_get_future() {
  boost::fibers::promise<int> p1;
  boost::fibers::future<int> f1 = p1.get_future();
  assert(f1.valid());
  bool thrown = false;
  try {
    f1 = p1.get_future();
  } catch (boost::fibers::future_already_retrieved const&) {
    thrown = true;
  }
  assert(thrown);
}

void test_get_future_ref() {
  boost::fibers::promise<int&> p1;
  boost::fibers::future<int&> f1 = p1.get_future();
  assert(f1.valid());
  bool thrown = false;
  try {
    f1 = p1.get_future();
  } catch (boost::fibers::future_already_retrieved const&) {
    thrown = true;
  }
  assert(thrown);
}

void test_get_future_move() {
  boost::fibers::promise<int> p1;
  boost::fibers::future<int> f1 = p1.get_future();
  assert(f1.valid());
  bool thrown = false;
  try {
    f1 = p1.get_future();
  } catch (boost::fibers::future_already_retrieved const&) {
    thrown = true;
  }
  assert(thrown);
  boost::fibers::promise<int> p2(boost::move(p1));
  thrown = false;
  try {
    f1 = p2.get_future();
  } catch (boost::fibers::future_already_retrieved const&) {
    thrown = true;
  }
  assert(thrown);
  thrown = false;
  try {
    f1 = p1.get_future();
  } catch (boost::fibers::promise_uninitialized const&) {
    thrown = true;
  }
  assert(thrown);
}

void doit() {
  create();
  create_ref();
  test_move();
  test_move_ref();
  test_move_void();
  test_swap();
  test_swap_ref();
  test_swap_void();
  test_get_future();
  test_get_future_ref();
  test_get_future_move();
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
