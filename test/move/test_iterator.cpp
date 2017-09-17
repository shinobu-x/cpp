#include <boost/move/iterator.hpp>
#include <boost/container/vector.hpp>

#include "hpp/movable.hpp"

auto main() -> decltype(0) {
  boost::container::vector<movable> v1(10);
  assert(!v1[0].moved());

  boost::container::vector<movable> v2(
    boost::make_move_iterator(v1.begin()),
    boost::make_move_iterator(v1.end()));

  assert(v1[0].moved());
  assert(!v2[0].moved());
  assert(v1.size() == 10);
  assert(v2.size() == 10);

  v1.assign(
    boost::make_move_iterator(v2.begin()),
    boost::make_move_iterator(v2.end()));

  assert(v2[0].moved());
  assert(!v1[0].moved());
  assert(v1.size() == 10);
  assert(v2.size() == 10);

  return 0;
}
