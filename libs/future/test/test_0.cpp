#include "../hpp/future.hpp"

int p1() {
  return 1;
}

int p2() {                   
  return 1;                  
}

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

void doit() {
  {
    boost::executor_ptr_type ex;
    boost::mutex mutex;
    boost::lock_guard<boost::mutex> lock_guard(mutex);
    boost::unique_lock<boost::mutex> unique_lock;

    boost::detail::shared_state_base base_type;
    auto r = base_type.get_executor();
    (void)r;
    base_type.set_executor_policy(ex);
    base_type.set_executor_policy(ex, lock_guard);
    base_type.set_executor_policy(ex, unique_lock);
  }
  {
    boost::promise<int> p;
    auto f = p.get_future();
    int i = 1;
    p.set_value(i);
    auto r = f.get();
  }
  {
    boost::promise<int> p;
    boost::future<int> f(p.get_future());
  }
  {
    int a = 10;
    auto f = boost::make_ready_future(a);
  }
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
