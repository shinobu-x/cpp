#define BOOST_THREAD_VERSION 4
#define BOOST_THREAD_PROVIDES_EXECUTORS

#include <boost/static_assert.hpp>
#include <boost/thread/executors/basic_thread_pool.hpp>
#include <boost/thread/future.hpp>

#include <cassert>

struct test_callback {
  boost::future<void> operator()(boost::future<void> f) const {
    assert(f.is_ready());
    f.wait();
    return boost::make_ready_future();
  }

  boost::future<void> operator()(boost::future<boost::future<void> > f) const {
    assert(f.is_ready());
    assert(f.get().is_ready());
    return boost::make_ready_future();
  }

  boost::future<void> operator()(boost::shared_future<void> f) const {
    assert(f.is_ready());
    f.wait();
    return boost::make_ready_future();
  }

  boost::future<void> operator()(boost::shared_future<boost::future<void> > f) {
    assert(f.is_ready());
    assert(f.get().is_ready());
    return boost::make_ready_future();
  }
};

void p() {}

auto main() -> decltype(0) {
  {
    auto f1 = boost::make_ready_future().then(test_callback());
    BOOST_STATIC_ASSERT(
      std::is_same<decltype(f1), boost::future<boost::future<void> > >::value);

    auto f2 = f1.unwrap();
    BOOST_STATIC_ASSERT(
      std::is_same<decltype(f2), boost::future<void> >::value);
    f2.wait();
  }
  {
    for (int i = 0; i < 5; ++i) {
      auto f1 = boost::make_ready_future().then(test_callback());
      BOOST_STATIC_ASSERT(
        std::is_same<decltype(f1),
        boost::future<boost::future<void> > >::value);

      auto f2 = f1.unwrap();
      BOOST_STATIC_ASSERT(
        std::is_same<decltype(f2), boost::future<void> >::value);
      f2.wait();
    }
  }
  {
    for (int i = 0; i < 5; ++i) {
      auto f1 = boost::make_ready_future().then(test_callback());
      BOOST_STATIC_ASSERT(
        std::is_same<decltype(f1),
        boost::future<boost::future<void> > >::value);
      auto f2 = f1.get();
    }
  }
  {
    auto f1 = boost::make_ready_future().then(test_callback());
    BOOST_STATIC_ASSERT(
      std::is_same<decltype(f1), boost::future<boost::future<void> > >::value);

/*    auto f3 = f1.then(test_callback());
    BOOST_STATIC_ASSERT(
      std::is_same<decltype(f2), boost::future<boost::future<void> > >::value);
    f2.wait(); */
  }
  {
    for (int i = 0; i < 5; ++i) {
      auto f1 = boost::make_ready_future().then(test_callback());
      BOOST_STATIC_ASSERT(
        std::is_same<decltype(f1),
        boost::future<boost::future<void> > >::value);

      auto f2 = f1.unwrap();
      BOOST_STATIC_ASSERT(
        std::is_same<decltype(f2), boost::future<void> >::value);

      auto f3 = f2.then(test_callback());
      BOOST_STATIC_ASSERT(
        std::is_same<decltype(f3),
        boost::future<boost::future<void> > >::value);
      f3.wait();

    }
  }

  {
    for (int i = 0; i < 5; ++i) {
      boost::basic_thread_pool executor;
      auto f1 = boost::make_ready_future().then(executor, test_callback());
      BOOST_STATIC_ASSERT(
        std::is_same<decltype(f1),
        boost::future<boost::future<void> > >::value);

/*      auto f2 = f1.then(executor, test_callback());
      BOOST_STATIC_ASSERT(
        std::is_same<decltype(f2),
        boost::future<boost::future<void> > >::value);
      f2.wait(); */
    }
  }
  return 0;
}
