#define BOOST_THREAD_VERSION 2 

#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/thread/future.hpp>

#include <cassert>
#include <memory>
#include <string>
#include <utility>

// #include "../thread/thread_move.hpp"
#include "../thread/thread_only.hpp"

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

struct X {
public:
  int i;

  BOOST_THREAD_MOVABLE_ONLY(X)
  X() : i(42) {}

  X(BOOST_THREAD_RV_REF(X) that)
    : i(BOOST_THREAD_RV(that).i) {
    BOOST_THREAD_RV(that).i = 0;
  }

  X& operator=(BOOST_THREAD_RV_REF(X) that) {
    i = BOOST_THREAD_RV(that).i;
    BOOST_THREAD_RV(that).i = 0;
    return *this;
  }

  ~X() {}
};

namespace boost {
  BOOST_THREAD_DCL_MOVABLE(X)
}

namespace {
int make_int() {
  return 1;
}

void set_promise_thread(boost::promise<int>* p) {
  p->set_value(1);
}

int thrown_runtime_error() {
  throw std::runtime_error("1");
}

struct exception {};

void set_promise_exception_thread(boost::promise<int>* p) {
  p->set_exception(boost::copy_exception(exception()));
}
} // namespace

// Test store value from thread
void test_1() {
  try {
    boost::promise<int> p;
    boost::unique_future<int> f(BOOST_THREAD_MAKE_RV_REF(p.get_future()));
    boost::thread(set_promise_thread, &p);
    int i = f.get();
    assert(i == 1);
    assert(f.is_ready());
    assert(f.has_value());
    assert(!f.has_exception());
    assert(f.get_state() == boost::future_state::ready);
  } catch (...) {}
}

auto main() -> decltype(0) {
  test_1();
  return 0;
}
