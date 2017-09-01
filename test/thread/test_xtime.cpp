#include <boost/thread/detail/config.hpp>
#include <boost/thread/xtime.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>

#include <cassert>

// Test xtime_cmp
void test_1() {
  boost::xtime xt1;
  boost::xtime xt2;
  boost::xtime cur;

  assert(
    (boost::xtime_get(&cur, boost::TIME_UTC_) ==
      static_cast<int>(boost::TIME_UTC_)));

  xt1 = xt2 = cur;
  xt1.nsec -= 1;
  xt2.nsec += 1;

  assert(boost::xtime_cmp(xt1, cur) < 0);
  assert(boost::xtime_cmp(xt2, cur) > 0);
  assert(boost::xtime_cmp(cur, cur) == 0);

  xt1 = xt2 = cur;
  xt1.sec -= 1;
  xt2.sec += 1;

  assert(boost::xtime_cmp(xt1, cur) < 0);
  assert(boost::xtime_cmp(xt2, cur) > 0);
  assert(boost::xtime_cmp(cur, cur) == 0);
}

// Test xtime_get
void test_2() {
  boost::xtime org;
  boost::xtime cur;
  boost::xtime old;

  assert(
    (boost::xtime_get(&org, boost::TIME_UTC_) ==
      static_cast<int>(boost::TIME_UTC_)));

  old = org;

  for (int x = 0; x < 100; ++x) {
    assert(
      (boost::xtime_get(&cur, boost::TIME_UTC_) ==
      static_cast<int>(boost::TIME_UTC_)));

    assert(boost::xtime_cmp(cur, org) >= 0);
    assert(boost::xtime_cmp(cur, old) >= 0);

    old = cur;
  }

}

// Test xtime_mutex backwards compatibility
void test_3() {
  boost::timed_mutex m;

  assert(m.timed_lock(boost::get_xtime(
    boost::get_system_time() + boost::posix_time::milliseconds(10))));

  m.unlock();

  boost::timed_mutex::scoped_timed_lock l(m, boost::get_xtime(
    boost::get_system_time() + boost::posix_time::milliseconds(10)));

  assert(l.owns_lock());

  if (l.owns_lock())
    l.unlock();

  assert(l.timed_lock(
    boost::get_xtime(
      boost::get_system_time() + boost::posix_time::milliseconds(10))));

  if (l.owns_lock())
    l.unlock();
}

bool predicate() {
  return false;
}

// Test xtime condition_variable backwards compatibility
void test_4() {
  boost::condition_variable cond;
  boost::condition_variable_any cond_any;
  boost::mutex m;

  boost::unique_lock<boost::mutex> l(m);
  cond.timed_wait(
    l, boost::get_xtime(
      boost::get_system_time() + boost::posix_time::milliseconds(10)));

  cond.timed_wait(
    l, boost::get_xtime(
      boost::get_system_time() + boost::posix_time::milliseconds(10)),
    predicate);


  cond_any.timed_wait(
    l, boost::get_xtime(
      boost::get_system_time() + boost::posix_time::milliseconds(10)));

  cond_any.timed_wait(
    l, boost::get_xtime(
      boost::get_system_time() + boost::posix_time::milliseconds(10)),
    predicate);
}
auto main() -> decltype(0) {
  test_1(); test_2(); test_3(); test_4();
  return 0;
}
