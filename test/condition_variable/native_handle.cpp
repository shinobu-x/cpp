#include <boost/thread/condition_variable.hpp>

#include <cassert>

auto main() -> decltype(0) {
  boost::condition_variable cv;
  boost::condition_variable::native_handle_type h = cv.native_handle();
  assert(h != 0);
  return 0;
}
