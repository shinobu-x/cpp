#include <iostream>
// #include <string>

enum optimization {
  LEVEL1,
  LEVEL2,
  LEVEL3,
  LEVEL4,
  LEVEL5
};

template <typename T, T LEVEL>
T doit(std::string& s, std::string& r0) {
  r0.clear();
  r0.reserve(s.length());
  std::string r1;
  std::string r2;  // Reseve storage space
  r2.reserve(s.length());
  for (T i = 0; i < s.length(); ++i) {
    if (s[i] >= 0x20) {
      switch (LEVEL) {
      case LEVEL1:
        r1 = r1 + s[i];  // With temporary storage
      case LEVEL2:
        r1 += s[i];  // Without temporary storage
      case LEVEL3:
        r2 += s[i];
      case LEVEL4:
        goto faster;
        break;
      }
    }
  }
faster:
{
  for (auto it = s.begin(), end = s.end(); it != end; ++it) {
    if (*it >= 0x20) {
      switch (LEVEL) {
      case LEVEL4:
        r2 += *it;
      case LEVEL5:
        r0 += *it;
      }
    }
  }
}
  return 0;
}

auto main() -> int
{
  std::string s = {"1234abcd"};
  std::string r;
  doit<int, 4>(s, r);
}
