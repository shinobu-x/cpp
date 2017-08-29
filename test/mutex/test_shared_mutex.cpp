#include <boost/thread/thread.hpp>
#include <boost/thread/xtime.hpp>

#include "hpp/shared_mutex_locking_thread.hpp"
#include "../utils/utils.hpp"

#define CHECK_LOCKED_VALUE_EQUAL(mutex_name, value, expected_value) { \
  boost::unique_lock<boost::mutex> lock(mutex_name); \
  assert(value == expected_value); \
}

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
  boost::unique_lock<boost::mutex> finish_lock(finish_mutex);

  try {
    for (unsigned i = 0; i < number_of_threads; ++i)
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
      boost::unique_lock<boost::mutex> l(unblocked_count_mutex);
      while (unblocked_count<number_of_threads)
        unblocked_condition.wait(l);
    }

    CHECK_LOCKED_VALUE_EQUAL(
      unblocked_count_mutex, unblocked_count, number_of_threads);

    finish_lock.unlock();
    threads.join_all();
  } catch (...) {
    threads.interrupt_all();
    threads.join_all();
    throw;
  }

  CHECK_LOCKED_VALUE_EQUAL(
    unblocked_count_mutex, unblocked_count, number_of_threads);

}

// Test only one writer permitted
void test_2() {
  unsigned const number_of_threads = 10;
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
        locking_thread<boost::unique_lock<shared_mutex> >(
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

    CHECK_LOCKED_VALUE_EQUAL(
      unblocked_count_mutex, unblocked_count, 1U);

    finish_lock.unlock();
    threads.join_all();
  } catch (...) {
    threads.interrupt_all();
    threads.join_all();
  }

  CHECK_LOCKED_VALUE_EQUAL(
    unblocked_count_mutex, unblocked_count, number_of_threads);

  CHECK_LOCKED_VALUE_EQUAL(
    unblocked_count_mutex, max_simultaneous_running, 1U);
}

// Test reader blocks writer
void test_3() {
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
      boost::unique_lock<boost::mutex> unblocked_lock(unblocked_count_mutex);
      while (unblocked_count < 1)
        unblocked_condition.wait(unblocked_lock);
    }

    CHECK_LOCKED_VALUE_EQUAL(
      unblocked_count_mutex, unblocked_count, 1U);

    threads.create_thread(
      locking_thread<boost::unique_lock<shared_mutex> >(
        rwm_mutex,
        unblocked_count,
        unblocked_count_mutex,
        unblocked_condition,
        finish_mutex,
        simultaneous_running_count,
        max_simultaneous_running));

    {
      boost::thread::sleep(delay(1));
    }

    CHECK_LOCKED_VALUE_EQUAL(
      unblocked_count_mutex, unblocked_count, 1U);

    finish_lock.unlock();
    threads.join_all();
  } catch (...) {
    threads.interrupt_all();
    threads.join_all();
    throw;
  }

  CHECK_LOCKED_VALUE_EQUAL(
    unblocked_count_mutex, unblocked_count, 2U);

  CHECK_LOCKED_VALUE_EQUAL(
    unblocked_count_mutex, max_simultaneous_running, 1U);

}

// Test unlocking writer unblocks all readers
void test_4() {
  boost::thread_group threads;
  shared_mutex rwm_mutex;
  boost::unique_lock<shared_mutex> write_lock(rwm_mutex);
  unsigned unblocked_count = 0;
  unsigned simultaneous_running_count = 0;
  unsigned max_simultaneous_running = 0;
  boost::mutex unblocked_count_mutex;
  boost::condition_variable unblocked_cond;
  boost::mutex finish_mutex;
  boost::unique_lock<boost::mutex> finish_lock(finish_mutex);
  unsigned const reader_count = 10;

  try {
    for (unsigned i = 0; i < reader_count; ++i)
      threads.create_thread(
        locking_thread<boost::shared_lock<shared_mutex> >(
          rwm_mutex,
          unblocked_count,
          unblocked_count_mutex,
          unblocked_cond,
          finish_mutex,
          simultaneous_running_count,
          max_simultaneous_running));

    {
      boost::thread::sleep(delay(1));
    }

    CHECK_LOCKED_VALUE_EQUAL(
      unblocked_count_mutex, unblocked_count, 0U);

    write_lock.unlock();

    {
      boost::unique_lock<boost::mutex> unblocked_lock(unblocked_count_mutex);
      while (unblocked_count < reader_count)
        unblocked_cond.wait(unblocked_lock);
    }

    CHECK_LOCKED_VALUE_EQUAL(
      unblocked_count_mutex, unblocked_count, reader_count);

    finish_lock.unlock();
    threads.join_all();
  } catch (...) {
    threads.interrupt_all();
    threads.join_all();
    throw;
  }

  CHECK_LOCKED_VALUE_EQUAL(
    unblocked_count_mutex, max_simultaneous_running, reader_count);
}

// Test unblocking last reader only unblocks one writer
void test_5() {
  boost::thread_group threads;

  shared_mutex rwm_mutex;
  unsigned unblocked_count = 0;
  unsigned simultaneous_running_readers = 0;
  unsigned max_simultaneous_readers = 0;
  unsigned simultaneous_writers = 0;
  unsigned max_simultaneous_writers = 0;
  boost::mutex 
auto main() -> decltype(0) {
  test_1(); test_2(); test_3(); test_4();
  return 0;
}
