#include "thread_move.hpp"

struct type_t {
  BOOST_THREAD_MOVABLE_ONLY(type_t)
  type_t() {}
};

auto main() -> decltype(0) {
  return 0;
}
