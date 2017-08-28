#include <boost/thread/thread.hpp>
#include <boost/thread/xtime.hpp>

#include "hpp/shared_mutex_locking_thread.hpp"
#include "../utils/utils.hpp"

// Test multiple readers
void test_1() {
  unsigned const number_of_threads = 10;
  boost::thread_group threads;
  boost::shared_mutex rwm_mutex;
  unsigned unblocked_count = 0;
  unsigned simultaneous_running_count = 0;
  unsigned max_simultaneous_running = 0;
  boost::mutex unblocked_count_mutex;
  boost::condition_variable unblocked_condition;
  boost::mutex finish_mutex;
  boost::unique_lock<boost::mutex> finish_lock(finish_mutex);

  try {
    for (unsigned i = 0; i < number_of_threads; ++i)
      threads.create_thread(
        locking_thread<boost::shared_lock<boost::shared_mutex> >(
          rwm_mutex,
          unblocked_count,
          unblocked_count_mutex,
          unblocked_condition,
          finish_mutex,
          simultaneous_running_count,
          max_simultaneous_running));

    {
      boost::unique_lock<boost::mutex> l(unblocked_count_mutex);
      while (unblocked_count<number_of_threads)
        unblocked_condition.wait(l);
    }

    finish_lock.unlock();
    threads.join_all();
  } catch (...) {
    threads.interrupt_all();
    threads.join_all();
    throw;
  }
}

// Test only one writer permitted
void test_2() {
  unsigned const number_of_threads = 10;
  boost::thread_group threads;
  boost::shared_mutex rwm_mutex;
  unsigned unblocked_count = 0;
  unsigned simultaneous_running_count = 0;
  unsigned max_simultaneous_running = 0;
  boost::mutex unblocked_count_mutex;
  boost::condition_variable unblocked_condition;
  boost::mutex finish_mutex;
  boost::unique_lock<boost::mutex> finish_lock(finish_mutex);

  try {
    for (unsigned i = 0; i < number_of_threads; ++i)
      threads.create_thread(
        locking_thread<boost::unique_lock<boost::shared_mutex> >(
          rwm_mutex,
          unblocked_count,
          unblocked_count_mutex,
          unblocked_condition,
          finish_mutex,
          simultaneous_running_count,
          max_simultaneous_running));

    {
      boost::thread::sleep(delay(2));
    }

    finish_lock.unlock();
    threads.join_all();
  } catch (...) {
    threads.interrupt_all();
    threads.join_all();
  }
}

auto main() -> decltype(0) {
  test_1();
  return 0;
}
