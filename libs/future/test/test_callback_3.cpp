#include <include/futures.hpp>
#include <boost/thread/executors/basic_thread_pool.hpp>

struct Callback {
  typedef boost::shared_future<void> result_type;
  typedef boost::shared_future<void> future_type;

  result_type operator()(future_type future) {
    return boost::make_shared_future();
  };

  result_type operator()(boost::shared_future<future_type> future) {
    return boost::make_shared_future();
  }
};

void doit() {
  boost::basic_thread_pool executor;
  auto f1 = boost::make_shared_future().then(executor, Callback());
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
