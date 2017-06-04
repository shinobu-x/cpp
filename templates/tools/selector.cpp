#include <iostream>

template <bool condition>
struct selector { };

typedef selector<true> true_type;
typedef selector<false> false_type;

template <bool condition, typename T>
void f(const T& x) {
  if (condition) std::cout << x << '\n';
  else std::cout << "False" << '\n';
}

template <bool condition, typename T>
void f(selector<condition>, const T& x) {
  if (condition) std::cout << x << '\n';
  else std::cout << "False" << '\n';
}

template <typename T>
T doit() {
  T a = 1;
  const selector<true> ON;
  const selector<false> OFF;
  f<true>(a);
  f<false>(a);
  f(selector<true>(), a);
  f(selector<false>(), a);
  f(true_type(), a);
  f(false_type(), a);
  f(ON, a);
  f(OFF, a);
}

auto main() -> int
{
  doit<int>();
  return 0;
}
