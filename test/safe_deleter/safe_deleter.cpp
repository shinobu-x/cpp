#include "safe_deleter.hpp"

struct test_data {};

auto main() -> decltype(0) {
  test_data* ptr;
  safe_deleter(ptr);
}
