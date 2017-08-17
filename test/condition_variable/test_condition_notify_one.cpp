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

// Test condition notify one wakes from timed wait
void test_5() {
  wait_for_flag data;

  boost::thread t(boost::bind(
    &wait_for_flag::timed_wait_without_predicate, &data));

  {
    boost::unique_lock<boost::mutex> l(data.m_);
    boost::this_thread::sleep_for(boost::chrono::seconds(2));
    data.flag_ = true;
    data.cond_.notify_one();
  }

  t.join();
  assert(data.woken_);
}

// Test condition notify one wakes from timed wait with predicate
void test_6() {
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

// Test multiple notify one calls wakes multiple threads
void test_7() {
  boost::thread t1(wait_for_cond_var_and_increase_count);
  boost::thread t2(wait_for_cond_var_and_increase_count);

  boost::this_thread::sleep_for(boost::chrono::seconds(1));
  multiple_wake_cond.notify_one();

  boost::thread t3(wait_for_cond_var_and_increase_count);

  boost::this_thread::sleep_for(boost::chrono::seconds(2));
  multiple_wake_cond.notify_one();
  multiple_wake_cond.notify_one();
  boost::this_thread::sleep_for(boost::chrono::seconds(3));

  {
    boost::unique_lock<boost::mutex> l(multiple_wake_mutex);
    assert(multiple_wake_count == 3);
  }

  t1.join();
  t2.join();
  t3.join();
}
auto main() -> decltype(0) {
  test_1(); test_2(); test_3(); test_4(); test_5(); test_6();
  return 0;
}
