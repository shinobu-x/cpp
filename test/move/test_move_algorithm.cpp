#include <boost/move/algorithm.hpp>
#include <boost/container/vector.hpp>

#include <cassert>

#include "hpp/movable.hpp"

auto main() -> decltype(0) {
  boost::container::vector<movable> v1(10);
  boost::container::vector<movable> v2(10);

  boost::move(v1.begin(), v1.end(), v2.begin());

  assert(v1[0].moved());
  assert(v2.size() == 10);
  assert(!v2[0].moved());

  boost::move_backward(v2.begin(), v2.end(), v1.end());

  assert(v2[1].moved());
  assert(v1.size() == 10);
  assert(!v1[1].moved());

  return 0;
}
