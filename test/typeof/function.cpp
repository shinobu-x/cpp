#include "typeof_t.hpp"

#include <iostream>

BOOST_STATIC_ASSERT(type_of::test<void()>::value);
BOOST_STATIC_ASSERT(type_of::test<double(bool)>::value);

auto main() -> decltype(0) {
  return 0;
}
