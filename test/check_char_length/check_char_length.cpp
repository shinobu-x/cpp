#include "check_char_length.hpp"

auto main() -> decltype(0) {
  const char buf[] = "abcdefghijklmnopqrstuvwxyz";
  check_char_length(buf);
}
