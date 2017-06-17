#include <iostream>
#include <string>

std::string blocking_v1(std::string s) {
  std::string r;
  /**
   *  # b = 0
   *  b = 0, i = 0, e = 8; b < 8; 1
   *  # b = 1
   *  b = 1, i = 1, e = 8; b < 8; 2
   *  # b = 2
   *  b = 2, i = 2, e = 8; b < 8; 3
   *  ...
   */
  for (size_t b = 0, i = b, e = s.length(); b < e; b = i + 1) {
    for (i = b; i < e; ++i)
      if (s[i] < 0x20)
        break;
    r = r + s.substr(b, i - b);
  }
  return r;
}

template <typename T>
T doit() {
  T s = "abc"; 
  blocking_v1(s);
}

auto main() -> int
{
  doit<std::string>();
  return 0;
}
