#define BOOST_THREAD_VERSION 4

#include <boost/thread/future.hpp>

#include <cassert>

struct test_data {
  test_data(int i) : i_(i) {}
  int i_;
};

auto main() -> decltype(0) {
  boost::promise<test_data> p;
  const test_data f(42);
  p.set_value(f);
  const boost::future<test_data>& cf_ref =
    boost::make_ready_future(test_data(42));
  assert(cf_ref.valid());

  return 0;
}
