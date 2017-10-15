#define BOOST_THREAD_VERSION 4

#include <boost/function.hpp>
#include <boost/thread/executors/executor.hpp>
#include <boost/thread/executors/basic_thread_pool.hpp>
#include <boost/thread/executors/scheduling_adaptor.hpp>
#include <boost/chrono/chrono_io.hpp>

#include <cassert>

void f(int i) {
  std::cout << i << "\n";
}

void do_test(const int n) {
  boost::executors::basic_thread_pool tp(4);
  boost::scheduling_adaptor<boost::executors::basic_thread_pool> sa(tp);

  for (int i = 0; i < n; ++i) {
    sa.submit_after(boost::bind(f, i), boost::chrono::seconds(i));
    sa.submit_after(boost::bind(f, i*2), boost::chrono::milliseconds(i*100));
  }
  boost::this_thread::sleep_for(boost::chrono::seconds(10));
}

auto main() -> decltype(0) {
  boost::chrono::steady_clock::time_point start =
    boost::chrono::steady_clock::now();
  do_test(5);
  boost::chrono::steady_clock::duration diff =
    boost::chrono::steady_clock::now() - start;
  assert(diff > boost::chrono::seconds(5));

  return 0;
}
