#include <boost/thread/mutex.hpp>
#include <boost/shared_ptr.hpp>

#include "thread_only.hpp"

#include <cassert>

void fake_function() {}

// Move thread from lvalue on construction
void test_1() {
  boost::thread src(fake_function);
  boost::thread::id src_id = src.get_id();
  boost::thread dest(boost::move(src));
  boost::thread::id dest_id = dest.get_id();
  assert(src_id == dest_id);
  assert(src.get_id() == boost::thread::id());
  dest.join();
}

// Move thread from lvalue on assignment
void test_2() {
  boost::thread src(fake_function);
  boost::thread::id src_id = src.get_id();
  boost::thread dest;
  dest = boost::move(src);
  boost::thread::id dest_id = dest.get_id();
  assert(src_id == dest_id);
  assert(src.get_id() == boost::thread::id());
  dest.join();
}

// Dispatch thread
boost::thread start_thread() {
  return boost::thread(fake_function);
}

// Move thread from rvalue on construct
void test_3() {
  boost::thread t(start_thread());
  assert(t.get_id() != boost::thread::id());
  t.join();
}

// Move thread from rvalue using explicit move
void test_4() {
  boost::thread t = start_thread();
  assert(t.get_id() != boost::thread::id());
  t.join();
}

// Move locked thread from lvalue on construction
void test_5() {
  boost::mutex m;
  boost::unique_lock<boost::mutex> l1(m);
  assert(l1.owns_lock());
  assert(l1.mutex() == &m);
  boost::unique_lock<boost::mutex> l2(boost::move(l1));
  assert(!l1.owns_lock());
  assert(!l1.mutex());
  assert(l2.owns_lock());
  assert(l2.mutex() == &m);
}

// Lock thread
boost::unique_lock<boost::mutex> get_lock(boost::mutex& m) {
  return boost::unique_lock<boost::mutex>(m);
}

// Move locked thread from rvalue on construction
void test_6() {
  boost::mutex m;
  boost::unique_lock<boost::mutex> l(get_lock(m));
  assert(l.owns_lock());
  assert(l.mutex() == &m);
}

namespace test_ns {
  template <typename T>
  T move(T& t) { return t.move(); }

  bool move_called = false;

  struct nc : public boost::shared_ptr<int> {
    nc() {}
    nc(nc&&) { move_called = true; }
    nc move() {
      move_called = true;
      return nc();
    }
  };
}

void test_7() {
  test_ns::nc src;
  test_ns::nc dest = boost::move(src);
  assert(test_ns::move_called);
}

auto main() -> decltype(0) {
  test_1(); test_2(); test_3(); test_4(); test_5(); test_6(); test_7();
  return 0;
}
