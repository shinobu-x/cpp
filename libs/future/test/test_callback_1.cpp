#include <include/futures.hpp>

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
    future.wait();
    return boost::make_ready_future();
  }

  result_type operator()(boost::shared_future<T> future) const {
    assert(future.is_ready());
    future.wait();
    return boost::make_ready_future();
  }

  result_type operator()(boost::shared_future<future_type> future) const {
    assert(future.is_ready());
    future.wait();
    return boost::make_ready_future();
  }
};

void doit() {
  {
    auto f = boost::make_ready_future().then(Callback<void>());
    static_assert(
      std::is_same<
        decltype(f),
        boost::future<boost::future<void> > >::value
    );
    f.wait();
  }
  {
    auto f1 = boost::make_ready_future().then(Callback<void>());
    static_assert(
      std::is_same<
        decltype(f1),
        boost::future<boost::future<void> > >::value
    );
    auto f2 = f1.unwrap();
    static_assert(
      std::is_same<
        decltype(f2),
        boost::future<void> >::value
    );
    f2.wait();
  }
  {
    auto f1 = boost::make_ready_future().then(Callback<void>());
    static_assert(
      std::is_same<
        decltype(f1),
        boost::future<boost::future<void> > >::value
    );
    auto f2 = f1.get();
    static_assert(
      std::is_same<
        decltype(f2),
        boost::future<void> >::value
    );
  }
  {
    auto f1 = boost::make_ready_future().then(Callback<void>());
    static_assert(
      std::is_same<
        decltype(f1),
        boost::future<boost::future<void> > >::value
    );
    auto f2 = f1.unwrap();
    static_assert(
      std::is_same<
        decltype(f2),
        boost::future<void> >::value
    );
    auto f3 = f2.then(Callback<void>());
    static_assert(
      std::is_same<
        decltype(f3),
        boost::future<boost::future<void> > >::value
    );
    f3.wait();
  }
  {
    auto f = boost::make_ready_future().
    then(Callback<void>()).
    unwrap().
    then(Callback<void>()).get();
    static_assert(
      std::is_same<
        decltype(f),
        boost::future<void> >::value
    );
    f.wait();
  }
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
