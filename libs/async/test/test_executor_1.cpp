#define BOOST_THREAD_PROVIDES_FUTURE
#define BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
#include <boost/thread/future.hpp>

auto f1() -> decltype(0) {
  std::cout << __func__ << '\n';
  boost::this_thread::sleep_for(boost::chrono::seconds(1));
  return 1;
}

auto f2(boost::future<int> f) -> decltype(0) {
  std::cout << __func__ << '\n';
  auto i = f.get();
  boost::this_thread::sleep_for(boost::chrono::seconds(1));
  return 2*i;
}

auto f3(boost::future<int> f) -> decltype(0) {
  std::cout << __func__ << '\n';
  auto i = f.get();
  boost::this_thread::sleep_for(boost::chrono::seconds(1));
  return 3*i;
}

auto f4(boost::future<int> f) -> decltype(0) {
  std::cout << __func__ << '\n';
  auto i = f.get();
  boost::this_thread::sleep_for(boost::chrono::seconds(1));
  return 4*i;
}

auto main() -> decltype(0) {
  {
    boost::future<int> fa = boost::async(boost::launch::async, &f1);
    assert(fa.valid());
    boost::future<int> fb = fa.then(&f2);
    assert(fb.valid());
    assert(!fa.valid());
    assert(fb.get() == 2);
  }
  {
    boost::future<int> fa = boost::async(boost::launch::async, &f1);
    assert(fa.valid());
    boost::future<int> fb = fa.then(&f2).then(&f3).then(&f4);
    assert(fb.valid());
    assert(!fa.valid());
    assert(fb.get() == 24);
  }
  {
    boost::future<int> fa = boost::async(boost::launch::async, &f1);
    assert(fa.valid());
    auto fb = fa.then(&f2);
    assert(fb.valid());
    assert(!fa.valid());
    assert(fb.get() == 2);
    fa = fb.then(&f4).then(&f3).then(&f2);
    assert(fa.valid());
    assert(!fb.valid());
    assert(fa.get() == 48);
  }
  return 0;
}
