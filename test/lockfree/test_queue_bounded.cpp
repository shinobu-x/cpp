#include <boost/lockfree/queue.hpp>
#include <boost/scoped_ptr.hpp>

#include "common.hpp"

auto main() -> decltype(0) {
  typedef queue_stress_test<true> test_type;
  boost::scoped_ptr<test_type> t(new test_type(4, 4));
  boost::lockfree::queue<long> q(128);
  t->run(q);
  return 0;
}
