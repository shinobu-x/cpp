#define BOOST_THREAD_VERSION 4

#include <boost/thread/future.hpp>

#include <cassert>

int p1() {
  boost::this_thread::sleep_for(boost::chrono::milliseconds(500));
  return 1;
}

int p2(boost::future<int> f) {
  int i = f.get();
  boost::this_thread::sleep_for(boost::chrono::milliseconds(500));
  return 2 * i;
}

int p3(boost::future<int> f) {
  assert(f.valid());
  boost::this_thread::sleep_for(boost::chrono::milliseconds(500));
}

void do_test_async_defer() {
  try {
    boost::future<int> f1 = boost::async(boost::launch::deferred, &p1);
    assert(f1.valid());

    boost::future<int> f2 = f1.then(&p2);
    assert(f2.valid());

    assert(!f1.valid());
  } catch (...) {}

  try {
    boost::future<int> f1 = boost::async(boost::launch::deferred, &p1);
    assert(f1.valid());
    boost::future<int> f2 = f1.then(&p2);
    assert(f2.valid());
    assert(!f1.valid());
    assert(f2.get() == 2);
  } catch (...) {}

  try {
    boost::future<int> f1 = boost::async(boost::launch::deferred, &p1);
    assert(f1.valid());
    boost::future<int> f2 = f1.then(&p3);
    assert(f2.valid());
    f2.wait();
  } catch (...) {}

  {
    boost::future<int> f2 = boost::async(boost::launch::deferred, p1).then(&p2);
    assert(f2.get() == 2);
  }

  {
    boost::future<int> f1 = boost::async(boost::launch::deferred, p1);
    boost::future<int> f1_then_f2 = f1.then(&p2);
    boost::future<int> f2 = f1_then_f2.then(&p2);
    assert(f2.get() == 4);
  }

  {
    boost::future<int> f1 = boost::async(boost::launch::deferred, p1);
    boost::future<int> f2 = f1.then(&p2).then(&p2);
    assert(f2.get() == 4);
  }

  {
    boost::future<int> f2 =
      boost::async(boost::launch::deferred, p1).then(&p2).then(&p2);
    assert(f2.get() == 4);
  }
}

auto main() -> decltype(0) {
  do_test_async_defer();
  return 0;
}
