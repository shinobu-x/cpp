#include <boost/thread/thread.hpp>
#include <boost/thread/xtime.hpp>

#include <cassert>

#include "./hpp/shared_mutex_locking_thread.hpp"
#include "../utils/utils.hpp"

#define CHECK_LOCKED_VALUE_EQUAL(mutex_name, value, expected_value) { \
  boost::unique_lock<boost::mutex> lock(mutex_name);                  \
  assert(value == expected_value);                                    \
}

auto main() -> decltype(0) {
  return 0;
}
