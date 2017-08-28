#include <boost/thread/thread.hpp>
#include <boost/thread/xtime.hpp>

#include "hpp/shared_mutex_locking_thread.hpp"

// Test multiple readers
void test_1() {
  unsigned const number_of_threads = 10;
  boost::thread_group threads;
  shared_mutex rwm_mutex;
  unsigned unblocked_count = 0;
  unsigned simultaneous_running_count = 0;
  unsigned max_simultaneous_running = 0;
  boost::mutex unblocked_count_mutex;
  boost::condition_variable unblocked_condition;
  boost::mutex finish_mutex;
  boost::unique_lock<boost::mutex> finsih_lock(finish_mutex);

//  try {
    for (unsigned i = 0; i < number_of_threads; ++i)
      threads.create_thread(
        locking_thread<boost::shared_lock<shared_mutex> >(
          rwm_mutex, unblocked_count, unblocked_count_mutex,
          unblocked_condition, finish_mutex, simultaneous_running_count,
          max_simultaneous_running));
/*
    {
      boost::unique_lock<boost::mutex> unblocked_count(unblocked_count_mutex);
      while (unblocked_count < number_of_threads)
        unblocked_condition.wait(unblocked_count_mutex);
    }

    CHECK_LOCKED_VALUE_EQUAL(
      unblocked_count_mutex, unblocked_count, number_of_threads);

    finish_lock.unlock();

    threads.join_all();
*/
//  } catch (...) {
/*    threads.interrupt_all();
    threads.join_all();
    throw;
*/
//  }

}

auto main() -> decltype(0) {
  return 0;
}
