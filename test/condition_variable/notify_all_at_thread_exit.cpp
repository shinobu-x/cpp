#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/thread.hpp>
#include <boost/chrono/chrono.hpp>

#include <cassert>

boost::condition_variable cv;
boost::mutex m;

void f() {
  boost::unique_lock<boost::mutex> l(m);
  boost::notify_all_at_thread_exit(cv, boost::move(l));
  boost::this_thread::sleep_for(boost::chrono::milliseconds(300));
}

auto main() -> decltype(0) {
  boost::unique_lock<boost::mutex> l(m);
  boost::thread(f).detach();
  boost::chrono::high_resolution_clock::time_point t0 = 
    boost::chrono::high_resolution_clock::now();
  cv.wait(l);
  boost::chrono::high_resolution_clock::time_point t1 =
    boost::chrono::high_resolution_clock::now();
  assert(t1 - t0 > boost::chrono::milliseconds(250));
  return 0;
}
