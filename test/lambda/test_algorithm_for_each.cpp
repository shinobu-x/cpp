#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/lambda/algorithm.hpp>

#include <cassert>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {
void for_each() {
  int a[10][20];
  int sum = 0;

  std::for_each(a, a + 10,
    boost::lambda::bind(
      boost::lambda::ll::for_each(),
      boost::lambda::_1,
      boost::lambda::_1 + 20,
      boost::lambda::protect(
        (
          boost::lambda::_1 = 
            boost::lambda::var(sum),
            ++boost::lambda::var(sum)))));

  sum = 0;

  std::for_each(a, a + 10,
    boost::lambda::bind(
      boost::lambda::ll::for_each(),
      boost::lambda::_1,
      boost::lambda::_1 + 20,
      boost::lambda::protect((
        sum += boost::lambda::_1))));

  assert(sum == (199 + 1) / 2 * 199);
}
} // namespace

auto main() -> decltype(0) {
  for_each();
  return 0;
}
