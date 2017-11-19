#include <cstring>
#include <string>
#include <iostream>

#include "test_1.hpp"

auto main() -> decltype(0) {
  const char buf[] = "abcdefghijklmnopqrstuvwxyz";
  check_length(buf);
}
