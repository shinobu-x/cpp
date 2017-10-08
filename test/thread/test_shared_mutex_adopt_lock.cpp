#include <boost/thread/lock_types.hpp>

#include <cassert>

#include "../mutex/hpp/shared_mutex.hpp"

auto main() -> decltype(0) {
  shared_mutex m;
  m.lock_shared();
  boost::shared_lock<shared_mutex> l(m, boost::adopt_lock);
  assert(l.mutex() == &m);
  assert(l.owns_lock() == true);
  return 0;
}
