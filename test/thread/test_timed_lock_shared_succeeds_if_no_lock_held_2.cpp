#include <boost/thread/thread.hpp>

#include <cassert>

#include "hpp/shared_mutex_locking_thread.hpp"

// Test timed lock shared succeeds if no lock held
struct test {
  void operator()() {
    shared_mutex rwm_mutex;
    boost::mutex finish_mutex;
    boost::mutex unblocked_mutex;
    boost::chrono::steady_clock::time_point const start =
      boost::chrono::steady_clock::now();
    boost::chrono::steady_clock::time_point const timeout1 =
      start + boost::chrono::milliseconds(500);
    boost::chrono::milliseconds const timeout_resolution(50);
    bool timed_lock_succeeded = rwm_mutex.try_lock_shared_until(timeout1);

    {
      assert(boost::chrono::steady_clock::now() < timeout1);
      assert(timed_lock_succeeded);
    }

    if (timed_lock_succeeded)
      rwm_mutex.unlock_shared();

    boost::chrono::milliseconds const wait_duration(500);
    boost::chrono::steady_clock::time_point const timeout2 =
      boost::chrono::steady_clock::now() + wait_duration;
    timed_lock_succeeded = rwm_mutex.try_lock_shared_for(wait_duration);

    {
      assert(boost::chrono::steady_clock::now() < timeout2);
      assert(timed_lock_succeeded);
    }

    if (timed_lock_succeeded)
      rwm_mutex.unlock_shared();
  }
};

void do_test() {
  test()();
}

auto main() -> decltype(0) {
  do_test();
  return 0;
}
