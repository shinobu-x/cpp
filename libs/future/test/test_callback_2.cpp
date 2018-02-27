#include <include/futures.hpp>
#include <boost/thread/executors/basic_thread_pool.hpp>

template <typename T>
struct Callback {
  typedef boost::future<T> result_type;
  typedef boost::future<T> future_type;

  result_type operator()(future_type future) const {
    std::cout << __LINE__ << '\n';
    assert(future.is_ready());
    future.wait();
    return boost::make_ready_future();
  }

  result_type operator()(boost::future<future_type> future) const {
    std::cout << __LINE__ << '\n';
    assert(future.is_ready());
    assert(future.get().is_ready());
    return boost::make_ready_future();
  }

  result_type operator()(boost::shared_future<T> future) const {
    std::cout << __LINE__ << '\n';
    assert(future.is_ready());
    future.wait();
    return boost::make_ready_future();
  }

  result_type operator()(boost::shared_future<future_type> future) const {
    std::cout << __LINE__ << '\n';
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
  {
    boost::basic_thread_pool executor(2);
    auto f1 = boost::make_ready_future().then(executor, Callback<void>());
    static_assert(
      std::is_same<
        decltype(f1),
        boost::future<boost::future<void> > >::value
    );
    std::cout << (int)f1.valid() << '\n';
    auto f2 = f1.unwrap();
    static_assert(
      std::is_same<
        decltype(f2),
        boost::future<void> >::value
    );
    std::cout << (int)f2.valid() << '\n';
    auto f3 = f2.then(executor, Callback<void>());
    static_assert(
      std::is_same<
        decltype(f3),
        boost::future<boost::future<void> > >::value
    );
    f3.wait();
  }
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
