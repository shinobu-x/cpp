#include <boost/thread/thread.hpp>
#include <boost/thread/xtime.hpp>

#include "hpp/shared_mutex_locking_thread.hpp"

void test_if_other_thread_has_upgrade_lock_try_lock_upgrade_return_false() {
  shared_mutex rwm_mutex;
  boost::mutex finish_mutex;
  boost::mutex unblocked_mutex;
  unsigned unblocked_count = 0;
  boost::unique_lock<boost::mutex> finish_lock(finish_mutex);
  boost::thread writer(
    simple_upgrade_thread(
      rwm_mutex,
      finish_mutex,
      unblocked_mutex,
      unblocked_count));

  boost::this_thread::sleep(boost::posix_time::seconds(1));

  assert(unblocked_count == 1u);

  bool const try_succeeded = rwm_mutex.try_lock_upgrade();

  assert(!try_succeeded);

  if (try_succeeded)
    rwm_mutex.unlock_upgrade();

  finish_lock.unlock();
  writer.join();
}

auto main() -> decltype(0) {
  test_if_other_thread_has_upgrade_lock_try_lock_upgrade_return_false();
  return 0;
}
