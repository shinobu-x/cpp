#include <boost/thread/thread.hpp>

#include "hpp/shared_mutex_locking_thread.hpp"
#include "../utils/utils.hpp"

void test_if_no_thread_has_upgrade_lock_try_lock_shared_return_true() {
  shared_mutex rwm_mutex;
  bool const try_succeeded = rwm_mutex.try_lock_shared();
  assert(try_succeeded);
  if (try_succeeded)
    rwm_mutex.unlock_shared();
}

auto main() -> decltype(0) {
  test_if_no_thread_has_upgrade_lock_try_lock_shared_return_true();
  return 0;
}
