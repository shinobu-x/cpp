#define BOOST_THREAD_VERSION 2
#define BOOST_THREAD_PROVIDES_INTERRUPTIONS

#include <boost/thread/detail/config.hpp>
#include <boost/thread/xtime.hpp>
#include <boost/bind.hpp>
#include <boost/ref.hpp>
#include <boost/utility.hpp>

#include "../thread/thread_only.hpp"
#include "../utils/utils.hpp"

#include <cassert>

int test_value;

void simple_thread() {
  test_value = 42;
}

void comparison_thread(boost::thread::id pid) {
  boost::thread::id const tid1 = boost::this_thread::get_id();
  assert(tid1 != pid);
  boost::thread::id const tid2 = boost::this_thread::get_id();
  assert(tid2 == pid);
  boost::thread::id const tid3 = boost::thread::id();
  assert(tid1 != tid3);
}

void interruption_point_thread(boost::mutex* m, bool* failed) {
  boost::unique_lock<boost::mutex> l(*m);
  boost::this_thread::interruption_point();
  *failed = true;
}

void disabled_interruption_point_thread(boost::mutex* m, bool* failed) {
  boost::unique_lock<boost::mutex> l(*m);
  boost::this_thread::disable_interruption disable;
  boost::this_thread::interruption_point();
  *failed = false;
}

struct non_copyable_functor : boost::noncopyable {
  unsigned value;

  non_copyable_functor() : boost::noncopyable(), value(0) {}

  void operator()() {
    value = 42;
  }
};

struct long_running_thread {
  boost::condition_variable _cond;
  boost::mutex _m;
  bool _done;
  long_running_thread() : _done(false) {}

  void operator()() {
    boost::unique_lock<boost::mutex> l(_m);
    while (!_done)
      _cond.wait(l);
  }
};

// Test sleep
void test_1() {
  boost::xtime t = delay(3);
  boost::thread::sleep(t);
  assert(in_range(t, 2));
}

// Test thread creation
void test_2() {
  test_value = 0;
  boost::thread t(&simple_thread);
  t.join();
  assert(test_value == 42);
}

// Test id comparison
void test_3() {
  boost::thread::id const this_id = boost::this_thread::get_id();
  boost::thread t(boost::bind(&comparison_thread, this_id));
  t.join();
}

// Test interruption thread at interruption point
void test_4() {
  boost::mutex m;
  bool failed = false;
  boost::unique_lock<boost::mutex> l(m);
  boost::thread t(boost::bind(&interruption_point_thread, &m, &failed));
  t.interrupt();
  l.unlock();
  t.join();
  assert(!failed);
}

// Test no interrupt thread if interrupts disabled at interruption point
void test_5() {
  boost::mutex m;
  bool failed = true;
  boost::unique_lock<boost::mutex> l(m);
  boost::thread t(
    boost::bind(&disabled_interruption_point_thread, &m, &failed));
  t.interrupt();
  l.unlock();
  t.join();
  assert(!failed);
}

// Test thread creattion through reference wrapper
void test_6() {
  non_copyable_functor f;
  boost::thread t(boost::ref(f));
  t.join();
  assert(f.value == 42u);
}

// Test timed join
void test_7() {
  long_running_thread f;
  boost::thread t(boost::ref(f));
  assert(t.joinable());
  boost::system_time tx = delay(3);
  bool const joined1 = t.timed_join(tx);
  assert(in_range(boost::get_xtime(tx), 2));
  assert(!joined1);
  assert(t.joinable());
  {
    boost::unique_lock<boost::mutex> l(f._m);
    f._done = true;
    f._cond.notify_one();
  }

  tx = delay(3);
  bool const joined2 = t.timed_join(tx);
  boost::system_time const now = boost::get_system_time();
  assert(tx > now);
  assert(joined2);
  assert(!t.joinable());
}
auto main() -> decltype(0) {
  test_1(); test_2(); test_3(); test_4(); test_5(); test_6(); test_7();
  return 0;
}
