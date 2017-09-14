#include <boost/thread/thread_only.hpp>

#include <cassert>

#include "hpp/shared_mutex_locking_thread.hpp"

// Test timed lock shared times out if write lock held
void test() {
  shared_mutex rwm_mutex;
  boost::mutex finish_mutex;
  boost::mutex unblocked_mutex;
  unsigned unblocked_count = 0;
  boost::unique_lock<boost::mutex> finish_lock(finish_mutex);
  {
    assert(finish_lock.owns_lock());
  }
  boost::thread writer(
    simple_writing_thread(
      rwm_mutex,
      finish_mutex,
      unblocked_mutex,
      unblocked_count));

  boost::this_thread::sleep_for(boost::chrono::seconds(1));

  {
    boost::unique_lock<boost::mutex> unblocked_lock(unblocked_mutex);
    assert(unblocked_lock.owns_lock());
    assert(unblocked_count == 1u);
  }

  boost::chrono::steady_clock::time_point const start =
    boost::chrono::steady_clock::now();
  boost::chrono::steady_clock::time_point const timeout1 = 
    start + boost::chrono::milliseconds(500);
  boost::chrono::milliseconds const timeout_resolution(50);
  bool timed_lock_succeeded = rwm_mutex.try_lock_shared_until(timeout1);

  {
    assert(
      (timeout1 - timeout_resolution) < boost::chrono::steady_clock::now());
    assert(!timed_lock_succeeded);
  }

  boost::chrono::milliseconds const wait_duration(500);
  boost::chrono::steady_clock::time_point const timeout2 =
    boost::chrono::steady_clock::now() + wait_duration;
  timed_lock_succeeded = rwm_mutex.try_lock_shared_for(wait_duration);

  {
    assert(
      (timeout2 - timeout_resolution < boost::chrono::steady_clock::now()));
    assert(!timed_lock_succeeded);
  }

  if (timed_lock_succeeded)
    rwm_mutex.unlock_shared();

  finish_lock.unlock();
  writer.join();
}

void do_test() {
  test();
}

auto main() -> decltype(0) {
  do_test();
  return 0;
}
