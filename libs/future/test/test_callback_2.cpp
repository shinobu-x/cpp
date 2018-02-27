#include <include/futures.hpp>
#include <boost/thread/executors/basic_thread_pool.hpp>

template <typename T>
struct Callback {
  typedef boost::future<T> result_type;
  typedef boost::future<T> future_type;

  result_type operator()(future_type future) const {
    assert(future.is_ready());
    future.wait();
    return boost::make_ready_future();
  }

  result_type operator()(boost::future<future_type> future) const {
    assert(future.is_ready());
    assert(future.get().is_ready());
    return boost::make_ready_future();
  }

  result_type operator()(boost::shared_future<T> future) const {
    assert(future.is_ready());
    future.wait();
    return boost::make_ready_future();
  }

  result_type operator()(boost::shared_future<future_type> future) const {
    assert(future.is_ready());
    assert(future.get().is_ready());
    return boost::make_ready_future();
  }
};

void doit() {
  {
    boost::basic_thread_pool executor;
    auto f1 = boost::make_ready_future().then(executor, Callback<void>());
    static_assert(
      std::is_same<
        decltype(f1),
        boost::future<boost::future<void> > >::value
    );
    auto f2 = f1.then(executor, Callback<void>());
    static_assert(
      std::is_same<
        decltype(f2),
        boost::future<boost::future<void> > >::value
    );
    f2.wait();
  }
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
