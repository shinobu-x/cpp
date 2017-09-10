#include <boost/thread/thread.hpp>
#include <boost/thread/xtime.hpp>

#include <cassert>
#include <iostream>

#include "hpp/shared_mutex_locking_thread.hpp"

void test_shared_mutex_multiple_readers() {
  unsigned const number_of_threads = 10;
  boost::thread_group readers;
  shared_mutex rwm_mutex;
  unsigned unblocked_count = 0;
  unsigned simultaneous_running_count = 0;
  unsigned max_simultaneous_running = 0;
  boost::mutex unblocked_count_mutex;
  boost::condition_variable unblocked_condition;
  boost::mutex finish_mutex;
  boost::unique_lock<boost::mutex> finish_lock(finish_mutex);

  try {
    for (unsigned i = 0; i < number_of_threads; ++i)
      readers.create_thread(
        locking_thread<boost::shared_lock<shared_mutex> >(
          rwm_mutex,
          unblocked_count,
          unblocked_count_mutex,
          unblocked_condition,
          finish_mutex,
          simultaneous_running_count,
          max_simultaneous_running));

    {
      boost::unique_lock<boost::mutex> lock(unblocked_count_mutex);

      while (unblocked_count < number_of_threads)
        unblocked_condition.wait(lock);
    }

    assert(unblocked_count == number_of_threads);

    finish_lock.unlock();

    readers.join_all();

  } catch (...) {
    readers.interrupt_all();
    readers.join_all();
    throw;
  }

  assert(max_simultaneous_running == number_of_threads);

}

auto main() -> decltype(0) {
  test_shared_mutex_multiple_readers();
  return 0;
}
