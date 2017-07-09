#include <iostream>

struct yes_type {
  enum { value = true };
};

struct no_type {
  enum { value = false };
};

template <typename T>
T doit() {
  yes_type yes;
  no_type no;

  if (yes.value)
    std::cout << "Yes" << '\n';

  if (no.value)
    std::cout << "No" << '\n';
}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
