#define BOOST_THREAD_VERSION 4
#define BOOST_RESULT_OF_USE_DECLTYPE
#define BOOST_THREAD_PROVIDES_EXECUTORS

#include <boost/static_assert.hpp>
#include <boost/thread/executors/basic_thread_pool.hpp>
#include <boost/thread/future.hpp>

#include <cassert>

struct test_callback {
  boost::future<void> operator()(boost::future<void> f) const {
    assert(f.is_ready());
    f.get();
    return boost::make_ready_future();
  }

  boost::future<void> operator()(boost::future<boost::future<void> > f) const {
    assert(f.is_ready());
    f.get();
    return boost::make_ready_future();
  }
};

auto main() -> decltype(0) {
  {
    boost::promise<void> p;
    boost::future<void> f(p.get_future());
    auto f1 = f.then(test_callback());
    BOOST_STATIC_ASSERT(
      std::is_same<decltype(f1), boost::future<boost::future<void> > >::value);
    auto f2 = f1.then(test_callback());
    BOOST_STATIC_ASSERT(
      std::is_same<decltype(f2), boost::future<boost::future<void> > >::value);
  }
  return 0;
}
