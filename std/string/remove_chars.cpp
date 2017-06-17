#include <iostream>

void remove_strings(char* d, char const** s, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    if (*s[i] > 0x20)
      *d++ = *s[i];
  }
  *d = 0;
}

template <typename T, T N>
T doit() {
  char const* s[] = {"a", "b", "c", "d"};
  char* d = (char*)(malloc(sizeof(s)));

  remove_strings(d, s, (sizeof(s)/sizeof(s[0])));
}

auto main() -> int
{
  doit<int, 10>();
  return 0;
}
