#include <boost/thread/thread_only.hpp>
#include <boost/thread/xtime.hpp>

#include <cassert>

#include "hpp/shared_mutex_locking_thread.hpp"

void test_timed_lock_shared_succeeds_if_no_lock_held() {
  shared_mutex rwm_mutex;
  boost::mutex finish_mutex;
  boost::mutex unblocked_mutex;

  boost::system_time const start = boost::get_system_time();
  boost::system_time const timeout1 =
    start + boost::posix_time::milliseconds(500);
  boost::posix_time::milliseconds const timeout_resolution(50);
  bool timed_lock_succeeded = rwm_mutex.timed_lock_shared(timeout1);
  assert(boost::get_system_time() < timeout1);
  assert(timed_lock_succeeded);

  if (timed_lock_succeeded)
    rwm_mutex.unlock_shared();

  boost::posix_time::milliseconds const wait_duration(500);
  boost::system_time const timeout2 =
    boost::get_system_time() + wait_duration;
  timed_lock_succeeded = rwm_mutex.timed_lock_shared(wait_duration);
  assert(boost::get_system_time() < timeout2);
  assert(timed_lock_succeeded);

  if (timed_lock_succeeded)
    rwm_mutex.unlock_shared();
}

void do_test_timed_lock_shared_succeeds_if_no_lock_held() {
  test_timed_lock_shared_succeeds_if_no_lock_held();
}

auto main() -> decltype(0) {
  do_test_timed_lock_shared_succeeds_if_no_lock_held();
  return 0;
}
