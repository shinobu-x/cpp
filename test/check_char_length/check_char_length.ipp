#include "check_char_length.hpp"

inline void check_char_length(const char* str) {
  std::size_t len = strlen(str);
  std::cout << len << '\n';
}
