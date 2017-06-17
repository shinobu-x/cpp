#include <iostream>
#include <string>

/**
 *  # b = 0
 *  b = 0, i = 0, e = 8; b < 8; 1
 *  # b = 1
 *  b = 1, i = 1, e = 8; b < 8; 2
 *  # b = 2
 *  b = 2, i = 2, e = 8; b < 8; 3
 *  ...
 */
template <typename T, typename U>
T blocking_v3(T& s) {
  T r;
  r.reserve(s.length());
  for (U b = 0, i = b; b < s.length(); b = i+1) {
    for (i = b; i < s.length(); ++i)
      if (s[i] < 0x20)
        break;
    r.append(s, b, i-b);
  }
}

template <typename T, typename U>
void doit() {
  T s = "abc";
  blocking_v3<T, U>(s);
}

auto main() -> int
{
  doit<std::string, size_t>();
  return 0;
}
