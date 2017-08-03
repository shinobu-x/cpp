#include <boost/thread/condition_variable.hpp>

auto main() -> decltype(0) {
  boost::condition_variable cv0;
  boost::condition_variable cv1;
  /* cv1 = cv0;  Don't compile */
  return 0;
}
