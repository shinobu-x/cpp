#include <cassert>

#include "../thread/thread_only.hpp"
#include "../thread/condition_test_common.hpp"
#include "../utils/utils.hpp"

unsigned const number_of_threads = 5;

// Test condition notify all wakes from wait
void test_1() {
  wait_for_flag data;
  boost::thread_group threads;

  try {
    for (unsigned i = 0; i < number_of_threads; ++i)
      threads.create_thread(boost::bind(
        &wait_for_flag::wait_without_predicate, &data));
    {
      boost::unique_lock<boost::mutex> l(data.m_);
      data.flag_ = true;
      data.cond_.notify_all();
      assert(data.woken_ == number_of_threads);
    }
  } catch (...) {
    threads.join_all();
    throw;
  }

}
/*
// Test condition notify all wakes from wait with predicate
void test_2() {
  wait_for_flag data;
  boost::thread_group threads;

  try {
    for (unsigned i = 0; i < number_of_threads; ++i)
      threads.create_thread(boost::bind(
        &wait_for_flag::wait_with_predicate, data));

    {
      boost::unique_lock<boost::mutex> l(data.m_);
      data.flag_ = true;
      data.cond_.notify_all();
    }

    threads.join_all();
    assert(data.woken_ == number_of_threads);
  } catch (...) {
    threads.join_all();
    throw;
  }
}

// Test condition notify all wakes from timed out
void test_3() {
  wait_for_flag data;
  boost::thread_group threads;

  try {
    for (unsigned i = 0; i < number_of_threads; ++i)
      threads.create_thread(boost::bind(
        &wait_for_flag::timed_wait_without_predicate, data));

    {
      boost::unique_lock<boost::mutex> l(data.m_);
      data.flag_ = true;
      data.cond_.notify_all();
    }

    threads.join_all();
    assert(data.woken_ == number_of_threads);
  } catch (...) {
    threads.join_all();
    throw;
  }
}
*/
auto main() -> decltype(0) {
  return 0;
}
