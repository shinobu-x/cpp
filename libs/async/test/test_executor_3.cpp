#define BOOST_THREAD_PROVIDES_FUTURE
#define BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
#include <boost/thread/future.hpp>

#include <iostream>

auto f1() -> decltype(0) {
  std::cout << __func__ << '\n';
  boost::this_thread::sleep_for(boost::chrono::seconds(1));
  return 0;
}

auto f2(boost::future<int> f) -> decltype(0) {
  std::cout << __func__ << '\n';
  return 0;
}

auto f3() -> decltype(0) {
  std::cout << __func__ << '\n';
  return 0;
}

auto main() -> decltype(0) {
  boost::future<int> fa = boost::async(boost::launch::deferred, &f1);
  boost::future<int> fb = fa.then(boost::launch::deferred, &f2);
//  auto r = fb.get();

  return 0;
}
