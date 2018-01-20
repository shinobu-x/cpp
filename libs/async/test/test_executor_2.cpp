#define BOOST_THREAD_PROVIDES_FUTURE
#define BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
#include <boost/thread/future.hpp>

#include <cassert>
#include <iostream>

auto f1() -> decltype(0) {
  std::cout << __func__ << '\n';
  boost::this_thread::sleep_for(boost::chrono::seconds(1));
  return 1;
}

auto f2(boost::future<int> f) -> decltype(0) {
  std::cout << __func__ << '\n';
  assert(f.valid());
  int i = f.get();
  boost::this_thread::sleep_for(boost::chrono::seconds(1));
  return i*2;
}

auto f3(boost::future<int> f) -> decltype(0) {
  std::cout << __func__ << '\n';
  assert(f.valid());
  int i = f.get();
  boost::this_thread::sleep_for(boost::chrono::seconds(1));
  return i*3;
}

auto main() -> decltype(0) {
  boost::future<int> fa = boost::async(boost::launch::async, &f1).then(
    boost::launch::async, &f2);
  assert(fa.valid());
  assert(2 == fa.get());
  boost::future<int> fb = fa.then(boost::launch::async, &f2);
  assert(fb.valid());
  assert(!fa.valid());
  assert(4 == fb.get());
}
