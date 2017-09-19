#define BOOST_THREAD_VERSION 4
#include <boost/thread/future.hpp>

void f(int) {}

auto main() -> decltype(0) {
  {
    boost::packaged_task<void(int)> task(f);
  }
  {
    boost::packaged_task<void(int)> task(f);
    task(0);
  }
  {
    boost::packaged_task<void(int)> task(f);
    int x = 0;
    task(x);
  }
  return 0;
}
