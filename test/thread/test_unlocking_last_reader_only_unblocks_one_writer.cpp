#include <boost/thread/thread.hpp>

#include "hpp/shared_mutex_locking_thread.hpp"
#include "../utils/utils.hpp"

void test_unlocking_last_reader_only_unblocks_one_writer() {
  boost::thread_group threads;
  shared_mutex rwm_mutex;
  unsigned unblocked_count = 0;
  unsigned simultaneous_running_readers = 0;
  unsigned max_simultaneous_readers = 0;
  unsigned simultaneous_running_writers = 0;
  unsigned max_simultaneous_writers = 0;
  boost::mutex unblocked_count_mutex;
  boost::condition_variable unblocked_condition;
  boost::mutex finish_reading_mutex;
  boost::mutex finish_writing_mutex;
  boost::unique_lock<boost::mutex> finish_reading_lock(finish_reading_mutex);
  boost::unique_lock<boost::mutex> finish_writing_lock(finish_writing_mutex);

  unsigned const reader_count = 10;
  unsigned const writer_count = 10;

  try {
    for (unsigned i = 0; i < reader_count; ++i)
      threads.create_thread(
        locking_thread<boost::shared_lock<shared_mutex> >(
          rwm_mutex,
          unblocked_count,
          unblocked_count_mutex,
          unblocked_condition,
          finish_reading_mutex,
          simultaneous_running_readers,
          max_simultaneous_readers));

    boost::thread::sleep(delay(1));

    for (unsigned i = 0; i < writer_count; ++i)
      threads.create_thread(
        locking_thread<boost::unique_lock<shared_mutex> >(
          rwm_mutex,
          unblocked_count,
          unblocked_count_mutex,
          unblocked_condition,
          finish_writing_mutex,
          simultaneous_running_writers,
          max_simultaneous_writers));

    {
      boost::unique_lock<boost::mutex> unblock(unblocked_count_mutex);
      while (unblocked_count < reader_count)
        unblocked_condition.wait(unblock);
    }

    boost::thread::sleep(delay(1));

    {
      boost::unique_lock<boost::mutex> unblock(unblocked_count_mutex);
      assert(unblocked_count == reader_count);
    }

    finish_reading_lock.unlock();

    {
      boost::unique_lock<boost::mutex> unblock(unblocked_count_mutex);
      while (unblocked_count < (reader_count + 1))
        unblocked_condition.wait(unblock);
    }

    {
      boost::unique_lock<boost::mutex> unblock(unblocked_count_mutex);
      assert(unblocked_count == (reader_count + 1));
    }

    finish_writing_lock.unlock();
    threads.join_all();
  } catch (...) {
    threads.interrupt_all();
    threads.join_all();
    throw;
  }

  {
    boost::unique_lock<boost::mutex> unblock(unblocked_count_mutex);
    assert(unblocked_count == (reader_count + writer_count));
  }

  {
    boost::unique_lock<boost::mutex> unblock(unblocked_count_mutex);
    assert(max_simultaneous_readers == reader_count);
  }

  {
    boost::unique_lock<boost::mutex> unblock(unblocked_count_mutex);
    assert(max_simultaneous_writers == 1u);
  }
}

auto main() -> decltype(0) {
  test_unlocking_last_reader_only_unblocks_one_writer();
  return 0;
}
