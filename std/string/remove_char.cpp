#include <iostream>
// #include <string>

template <bool condition>
struct selector {};

typedef selector<true> true_type;
typedef selector<false> false_type;

template <typename T, bool is_optimize>
T doit(std::string& s) {
  std::string r;
  for (T i = 0; i < s.length(); ++i) {
    if (s[i] >= 0x20) {
      if (is_optimize)
        r += s[i];  // Without temporary storage
      else if (!is_optimize)
        r = r + s[i];  // With temporary storage
    }
  }
}

auto main() -> int
{
  std::string s = {"1234abcd"};
  doit<int, true>(s);
}
