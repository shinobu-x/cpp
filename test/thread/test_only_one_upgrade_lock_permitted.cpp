#include <boost/thread/thread.hpp>
#include <boost/thread/xtime.hpp>

#include "hpp/shared_mutex_locking_thread.hpp"
#include "../utils/utils.hpp"

void test_only_one_upgrade_lock_permitted() {
  unsigned const number_of_threads = 2;
  boost::thread_group threads;
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
      threads.create_thread(
        locking_thread<boost::upgrade_lock<shared_mutex> >(
          rwm_mutex,
          unblocked_count,
          unblocked_count_mutex,
          unblocked_condition,
          finish_mutex,
          simultaneous_running_count,
          max_simultaneous_running));

    boost::thread::sleep(delay(1));

    assert(unblocked_count == 1U);

    finish_lock.unlock();

    threads.join_all();

  } catch (...) {
    threads.interrupt_all();
    threads.join_all();
    throw;
  }
}

auto main() -> decltype(0) {
  test_only_one_upgrade_lock_permitted();
  return 0;
}
