#include <boost/thread/thread.hpp>

#include "hpp/shared_mutex_locking_thread.hpp"
#include "../utils/utils.hpp"

void test_if_other_thread_has_shared_lock_try_lock_upgrade_return_true() {
  shared_mutex rwm_mutex;
  boost::mutex finish_mutex;
  boost::mutex unblocked_mutex;
  unsigned unblocked_count = 0;
  boost::unique_lock<boost::mutex> finish_lock(finish_mutex);
  boost::thread writer(
    simple_reading_thread(
      rwm_mutex,
      finish_mutex,
      unblocked_mutex,
      unblocked_count));

  boost::thread::sleep(delay(1));

  {
    boost::unique_lock<boost::mutex> unblock(unblocked_mutex);
    assert(unblocked_count == 0);
  }

  bool const try_succeeded = rwm_mutex.try_lock_upgrade();
  assert(try_succeeded);

  if (try_succeeded)
    rwm_mutex.unlock_upgrade();

  finish_lock.unlock();
  writer.join();

  {
    boost::unique_lock<boost::mutex> unblock(unblocked_mutex);
    assert(unblocked_count == 1u);
  }
}

auto main() -> decltype(0) {
  test_if_other_thread_has_shared_lock_try_lock_upgrade_return_true();
  return 0;
}
