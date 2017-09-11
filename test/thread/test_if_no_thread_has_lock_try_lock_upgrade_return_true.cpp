#include <boost/thread/thread.hpp>

#include "hpp/shared_mutex_locking_thread.hpp"

void test_if_no_thread_has_lock_try_lock_upgrade_return_true() {
  shared_mutex rwm_mutex;
  bool const try_succeeded = rwm_mutex.try_lock_upgrade();
  assert(try_succeeded);
  if (try_succeeded)
    rwm_mutex.unlock_upgrade();
}

auto main() -> decltype(0) {
  test_if_no_thread_has_lock_try_lock_upgrade_return_true();
  return 0;
}
