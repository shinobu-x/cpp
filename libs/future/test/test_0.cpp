#include "../hpp/future.hpp"

template <typename T>
struct test_callback {
  boost::future<T> operator()(boost::future<T> f) const {
    assert(f.is_ready());
    f.get();
    return boost::make_ready_future();
  }

  boost::future<T> operator()(boost::future<boost::future<T> > f) const {
    assert(f.is_ready());
    f.get();
    return boost::make_ready_future();
  }
};

void doit() {
  {
    boost::promise<int> p;
    auto f = p.get_future();
    int i = 1;
    p.set_value(i);
    auto r = f.get();
  }
  {
    boost::promise<void> p;
    boost::future<void> f = p.get_future();
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
