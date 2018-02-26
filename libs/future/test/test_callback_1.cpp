#include <include/futures.hpp>

struct Callback {
  typedef boost::future<void> result_type;
  typedef boost::future<void> future_type;

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

  result_type operator()(boost::shared_future<void> future) const {
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
    auto f = boost::make_ready_future().then(Callback());
    static_assert(std::is_same<
      decltype(f),
      boost::future<boost::future<void> > >::value);
    f.wait();
  }
  {
    auto f1 = boost::make_ready_future().then(Callback());
    static_assert(std::is_same<
      decltype(f1),
      boost::future<boost::future<void> > >::value);
    auto f2 = f1.unwrap();
    static_assert(std::is_same<
      decltype(f2),
      boost::future<void> >::value);
    f2.wait();
  }
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
