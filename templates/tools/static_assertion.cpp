#include <iostream>

template <bool condition>
struct static_assertion;

template < >
struct static_assertion<true> {
  static_assertion() {}

  template <typename T>
  static_assertion(T) {}
};

template < >
struct static_assertion<false>;

struct error_CHAR_IS_UNSIGNED {};

template <typename T1, typename T2>
void doit() {
  const static_assertion<sizeof(T1) != 8> ASSERT1("invalid");
  const static_assertion<(T2(255) > 0)> ASSERT2(error_CHAR_IS_UNSIGNED());
}

auto main() -> int
{
  doit<double, char>();
  return 0;
}
