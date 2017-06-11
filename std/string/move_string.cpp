#include <iostream>
// #include <string>

enum optimization {
  LEVEL1,
  LEVEL2,
  LEVEL3,
  LEVEL4
};

template <typename T, T LEVEL>
T doit(std::string& s) {
  std::string r1;
  std::string r2;  // Reseve storage space
  r2.reserve(s.length());
  for (T i = 0; i < s.length(); ++i) {
    if (s[i] >= 0x20) {
      switch (LEVEL) {
      case LEVEL1:
        r1 = r1 + s[i];  // With temporary storage
        break;
      case LEVEL2:
        r1 += s[i];  // Without temporary storage
        break;
      case LEVEL3:
        r2 += s[i];
        break;
      case LEVEL4:
        goto fast;
        break;
      }
    }
  }
fast:
{
  for (auto it = s.begin(), end = s.end(); it != end; ++it)
    if (*it >= 0x20)
      r2 += *it;
}
  return 0;
}

auto main() -> int
{
  std::string s = {"1234abcd"};
  doit<int, 3>(s);
}
