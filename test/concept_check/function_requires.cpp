#include <boost/concept_check.hpp>

struct dummy {};

auto main() -> decltype(0) {
  boost::function_requires<boost::EqualityComparable<dummy> >();

  return 0;
}
