#include <boost/thread/thread_only.hpp>
#include <boost/thread/xtime.hpp>

#include <cassert>

#include "./hpp/shared_mutex_locking_thread.hpp"
#include "../utils/utils.hpp"

#define CHECK_LOCKED_VALUE_EQUAL(mutex_name, value, expected_value) { \
  boost::unique_lock<boost::mutex> lock(mutex_name);                  \
  assert(value == expected_value);                                    \
}

// Test timed lock shared times out if write lock held
void test_1() {
  shared_mutex rwm_mutex;
  boost::mutex finish_mutex;
  boost::mutex unblocked_mutex;
  unsigned unblocked_count = 0;
  boost::unique_lock<boost::mutex> finish_lock(finish_mutex);
  boost::thread writer(
    simple_writing_thread(
      rwm_mutex, finish_mutex, unblocked_mutex, unblocked_count));
  boost::thread::sleep(delay());

  CHECK_LOCKED_VALUE_EQUAL(unblocked_mutex, unblocked_count, 1u);

  boost::system_time const start =
    boost::get_system_time();
  boost::system_time const timeout1 =
    start + boost::posix_time::milliseconds(500);
  boost::posix_time::milliseconds const timeout_resolution(50);
  bool timed_lock_succeeded =
    rwm_mutex.timed_lock_shared(timeout1);

  assert(
    ((timeout1 - timeout_resolution) < boost::get_system_time()));
  assert(!timed_lock_succeeded);

  if (timed_lock_succeeded)
    rwm_mutex.unlock_shared();

  boost::posix_time::milliseconds const wait_duration(500);
  boost::system_time const timeout2 =
    boost::get_system_time() + wait_duration;
  timed_lock_succeeded = rwm_mutex.timed_lock_shared(wait_duration);

  assert(
    ((timeout2 - timeout_resolution) < boost::get_system_time()));
  assert(!timed_lock_succeeded);

  if (timed_lock_succeeded)
    rwm_mutex.unlock_shared();

  finish_lock.unlock();
  writer.join(); 
}

auto main() -> decltype(0) {
  test_1();
  return 0;
}
