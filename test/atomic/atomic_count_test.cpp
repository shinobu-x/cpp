#include <boost/detail/atomic_count.hpp>

#include <cassert>

auto main() -> decltype(0) {
  boost::detail::atomic_count n(4);
  assert(n == 4);
  ++n;
  assert(n == 5);
  assert(--n == 4);

  boost::detail::atomic_count m(0);
  assert(m == 0);
  ++m;
  assert(m == 1);
  ++m;
  assert(m == 2);
  assert(--m == 1);
  assert(--m == 0); 

  boost::detail::atomic_count a(4);
  assert(a == 4);
  assert(++a == 5);
  assert(++a == 6);
  assert(a == 6);

  boost::detail::atomic_count b(0);
  assert(b == 0);
  assert(++b == 1);
  assert(++b == 2);
  assert(b == 2);
  assert(--b == 1);
  assert(--b == 0);
  assert(--b == -1);
  assert(--b == -2);
  assert(b == -2);
  assert(++b == -1);
  assert(++b == 0);
  assert(b == 0);

  return 0;
}
