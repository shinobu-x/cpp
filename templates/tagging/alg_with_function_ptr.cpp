#include <cmath>
#include <iostream>

template <typename T, T VALUE>
struct static_parameter {};

template <typename T, T VALUE>
struct static_value : static_parameter<T, VALUE> {
  const static T value = VALUE;
};

enum algorithm_tag_t {
  NAIVE,
  PRECISE
};

inline static_value<algorithm_tag_t, NAIVE> naive_algorithm_tag() {}

inline static_value<algorithm_tag_t, PRECISE> precise_algorithm_tag() {}

typedef static_value<algorithm_tag_t, NAIVE> (*naive_algorithm_tag_t)();

typedef static_value<algorithm_tag_t, PRECISE> (*precise_algorithm_tag_t)();

namespace algorithm_tag {
  inline static_value<algorithm_tag_t, NAIVE> naive() {}

  inline static_value<algorithm_tag_t, PRECISE> precise() {}
}

template <typename T>
T doit() {

}

auto main() -> int
{
  return 0;
}
