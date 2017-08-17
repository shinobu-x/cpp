#include <boost/thread/detail/config.hpp>
#include <boost/chrono/chrono.hpp>

#include "../thread/thread_only.hpp"
#include "../thread/condition_test_common.hpp"

namespace {
  boost::mutex multiple_wake_mutex;
  boost::condition_variable multiple_wake_cond;
  unsigned multiple_wake_count = 0;

  void wait_for_cond_var_and_increase_count() {
    boost::unique_lock<boost::mutex> l(multiple_wake_mutex);
    multiple_wake_cond.wait(l);
    ++multiple_wake_count;
  }
}

// Test condition notify one wakes from wait
void test_1() {
  wait_for_flag data;

  boost::thread t(boost::bind(
    &wait_for_flag::wait_without_predicate, &data));

  {
    boost::unique_lock<boost::mutex> l(data.m_);
    data.flag_ = true;
    data.cond_.notify_one();
  }

  t.join();
  assert(data.woken_);
}

// Test condition notify one wakes from wait with predicate
void test_2() {
  wait_for_flag data;

  boost::thread t(boost::bind(
    &wait_for_flag::wait_with_predicate, &data));

  {
    boost::unique_lock<boost::mutex> l(data.m_);
    boost::this_thread::sleep_for(boost::chrono::seconds(3));
    data.flag_ = true;
    data.cond_.notify_one();
  }

  t.join();
  assert(data.woken_);
}

// Test condition notify one wakes from timed wait
void test_3() {
  wait_for_flag data;

  boost::thread t(boost::bind(
    &wait_for_flag::timed_wait_with_predicate, &data));

  {
    boost::unique_lock<boost::mutex> l(data.m_);
    data.flag_ = true;
    data.cond_.notify_one();
  }

  t.join();
  assert(data.woken_);
}

// Test condition notify one wakes from relative timed wait with predicate
void test_4() {
  wait_for_flag data;

  boost::thread t(boost::bind(
    &wait_for_flag::relative_timed_wait_with_predicate, &data));

  {
    boost::unique_lock<boost::mutex> l(data.m_);
    data.flag_ = true;
    data.cond_.notify_one();
  }

  t.join();
  assert(data.woken_);
}

auto main() -> decltype(0) {
  test_1(); test_2(); test_3();
  return 0;
}
