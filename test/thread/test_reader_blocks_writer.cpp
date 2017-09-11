#include <boost/thread/thread.hpp>

#include "hpp/shared_mutex_locking_thread.hpp"
#include "../utils/utils.hpp"

void test_reader_blocks_writer() {
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
    threads.create_thread(
      locking_thread<boost::shared_lock<shared_mutex> >(
        rwm_mutex,
        unblocked_count,
        unblocked_count_mutex,
        unblocked_condition,
        finish_mutex,
        simultaneous_running_count,
        max_simultaneous_running));

    {
      boost::unique_lock<boost::mutex> unblock(unblocked_count_mutex);
      while (unblocked_count < 1)
        unblocked_condition.wait(unblock);
    }

    {
      boost::unique_lock<boost::mutex> unblock(unblocked_count_mutex);
      assert(unblocked_count == 1U);
    }

    threads.create_thread(
      locking_thread<boost::unique_lock<shared_mutex> >(
        rwm_mutex,
        unblocked_count,
        unblocked_count_mutex,
        unblocked_condition,
        finish_mutex,
        simultaneous_running_count,
        max_simultaneous_running));

    boost::thread::sleep(delay(1));

    {
      boost::unique_lock<boost::mutex> unblock(unblocked_count_mutex);
      assert(unblocked_count == 1U);
    }

    finish_lock.unlock();
    threads.join_all();
  } catch (...) {
    threads.interrupt_all();
    threads.join_all();
    throw;
  }

  {
    boost::unique_lock<boost::mutex> unblock(unblocked_count_mutex);
    assert(unblocked_count == 2U);
  }

  {
    boost::unique_lock<boost::mutex> unblock(unblocked_count_mutex);
    assert(max_simultaneous_running == 1u);
  }
}

auto main() -> decltype(0) {
  test_reader_blocks_writer();
  return 0;
}
