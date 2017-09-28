#include <boost/shared_ptr.hpp>
#include <boost/functional/hash.hpp>

#include <cassert>

auto main() -> decltype(0) {
  boost::hash<boost::shared_ptr<int> > hasher;
  boost::shared_ptr<int> p1;
  boost::shared_ptr<int> p2(p1);
  boost::shared_ptr<int> p3(new int);
  boost::shared_ptr<int> p4(p3);
  boost::shared_ptr<int> p5(new int);

  assert(p1 == p2);
  assert(hasher(p1) == hasher(p2));

  assert(p1 != p3);
  assert(hasher(p1) != hasher(p3));

  assert(p3 == p4);
  assert(hasher(p3) == hasher(p4));

  assert(p4 != p5);
  assert(hasher(p4) != hasher(p5));

  return 0;
}
