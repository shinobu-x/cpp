#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/thread/future.hpp>

#include <cassert>
#include <iostream>
#include <string>
#include <utility>
#include <memory>

#include "./hpp/thread_only.hpp"
#include "../macro/config.hpp"

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
template <typename T>
typename boost::remove_reference<T>::type&& cast_to_rval(T&& t) {
  return static_cast<typename boost::remove_reference<T>::type&&>(t);
}
#else
#if defined BOOST_THREAD_USES_MOVE
template <typename T>
boost::rv<T>& cast_to_rval(T& t) {
  return boost::move(t);
}
#else
template <typename T>
boost::detail::thread_move_t<T> cast_to_rval(T& t) {
  return boost::move(t);
}
#endif
#endif

struct type_t {
  int i;

  BOOST_THREAD_MOVABLE_ONLY(type_t);

  type_t() : i(42) {}

  type_t(BOOST_THREAD_RV_REF(type_t) that) : i(BOOST_THREAD_RV(that).i) {
    BOOST_THREAD_RV(that).i = 0;
  }

  type_t& operator=(BOOST_THREAD_RV_REF(type_t) that) {
    i = BOOST_THREAD_RV(that).i;
    BOOST_THREAD_RV(that).i = 0;
    return *this;
  }

  ~type_t() {}
};

namespace boost {
  BOOST_THREAD_DCL_MOVABLE(type_t);
}

int make_int() {
  return 42;
}

int throw_runtime_error() {
  throw std::runtime_error("42");
}

void set_promise_thread(boost::promise<int>* p) {
  p->set_value(make_int());
}

struct exception {};

void set_promise_exception_thread(boost::promise<int>* p) {
  p->set_exception(boost::copy_exception(exception()));
}

// Test store value from thread
void test_1() {
LOG;
  try {
    boost::promise<int> p;
    boost::unique_future<int> f(BOOST_THREAD_MAKE_RV_REF(p.get_future()));
    boost::thread(set_promise_thread, &p);
    int j = f.get();

    assert(j == 42);
    assert(f.is_ready());
    assert(f.has_value());
    assert(!f.has_exception());
    assert(f.get_state() == boost::future_state::ready);
  } catch (...) {
    assert(false && "Exception thrown");
  }
}

// Test store exception
void test_2() {
LOG;
  boost::promise<int> p;
  boost::unique_future<int> f(BOOST_THREAD_MAKE_RV_REF(p.get_future()));
  boost::thread(set_promise_exception_thread, &p);

  try {
    f.get();
    assert(false);
  } catch (...) {
    assert(true);
  }

  assert(f.is_ready());
  assert(!f.has_value());
  assert(f.has_exception());
  assert(f.get_state() == boost::future_state::ready);
}

// Test waiting future
void test_3() {
LOG;
  boost::promise<int> p;
  boost::unique_future<int> f;
  f = BOOST_THREAD_MAKE_RV_REF(p.get_future());
  int i = 0;
  assert(!f.is_ready());
  assert(!f.has_value());
  assert(!f.has_exception());
  assert(f.get_state() == boost::future_state::waiting);
  assert(i == 0);
}

// Test initial state
void test_4() {
LOG;
  boost::unique_future<int> f;
  assert(!f.is_ready());
  assert(!f.has_value());
  assert(!f.has_exception());
  assert(f.get_state() == boost::future_state::uninitialized);
  int i;

  try {
    i = f.get();
    (void)i;
    assert(false);
  } catch (...) {
    assert(true);
  }
}

// Test cannot get future twice
void test_5() {
LOG;
  boost::promise<int> p;
  BOOST_THREAD_MAKE_RV_REF(p.get_future());

  try {
    p.get_future();
    assert(false);
  } catch (...) {
    assert(true);
  }
}

// Test set value updates future state
void test_6() {
LOG;
  boost::promise<int> p;
  boost::unique_future<int> f;
  f = BOOST_THREAD_MAKE_RV_REF(p.get_future());

  p.set_value(make_int());

  assert(f.is_ready());
  assert(f.has_value());
  assert(!f.has_exception());
  assert(f.get_state() == boost::future_state::ready);
}

// Test set value can be retrieved
void test_7() {
LOG;
  boost::promise<int> p;
  boost::unique_future<int> f;
  f = BOOST_THREAD_MAKE_RV_REF(p.get_future());

  p.set_value(make_int());

  int x = f.get();

  assert(x == 42);
  assert(f.is_ready());
  assert(f.has_value());
  assert(!f.has_exception());
  assert(f.get_state() == boost::future_state::ready);
}

auto main() -> decltype(0) {
  test_1(); test_2(); test_3(); test_4(); test_5();
  return 0;
}
