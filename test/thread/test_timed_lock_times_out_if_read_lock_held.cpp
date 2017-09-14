#include <boost/thread/thread_only.hpp>
#include <boost/thread/xtime.hpp>

#include <cassert>

#include "hpp/shared_mutex_locking_thread.hpp"

struct test_timed_lock_times_out_if_read_lock_held {
  void operator()() {
    shared_mutex rwm_mutex;
    boost::mutex finish_mutex;
    boost::mutex unblocked_mutex;
    unsigned unblocked_count = 0;
    boost::unique_lock<boost::mutex> finish_lock(finish_mutex);
    boost::thread reader(
      simple_reading_thread(
        rwm_mutex,
        finish_mutex,
        unblocked_mutex,
        unblocked_count));
    boost::system_time const start1 = boost::get_system_time();
    boost::system_time const timeout1 =
      start1 + boost::posix_time::seconds(2);
    boost::thread::sleep(timeout1);

    {
      boost::unique_lock<boost::mutex> unblocked_lock(unblocked_mutex);
      assert(unblocked_count == 1u);
    }

    boost::system_time const start2 = boost::get_system_time();
    boost::system_time const timeout2 =
      start2 + boost::posix_time::milliseconds(500);
    boost::posix_time::milliseconds const timeout_resolution(50);
    bool timed_lock_succeeded = rwm_mutex.timed_lock(timeout2);

    {
      assert((timeout2 - timeout_resolution) < boost::get_system_time());
      assert(!timed_lock_succeeded);
    }

    if (timed_lock_succeeded)
      rwm_mutex.unlock();

    boost::posix_time::milliseconds const wait_duration(500);
    boost::system_time const timeout3 =
      boost::get_system_time() + wait_duration;
    timed_lock_succeeded = rwm_mutex.timed_lock(wait_duration);

    {
      assert((timeout3 - timeout_resolution) < boost::get_system_time());
      assert(!timed_lock_succeeded);
    }

    if (timed_lock_succeeded)
      rwm_mutex.unlock();

    finish_lock.unlock();
    reader.join();
  }
};

void do_test_timed_lock_times_out_if_read_lock_held() {
  test_timed_lock_times_out_if_read_lock_held()();
}

auto main() -> decltype(0) {
  do_test_timed_lock_times_out_if_read_lock_held();
  return 0;
}
