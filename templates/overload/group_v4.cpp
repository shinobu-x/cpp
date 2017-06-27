#include <iostream>

struct maths {
  template <typename T>
  inline static T abs(const T x) {
    return x<0 ? -x : x;
  }

  inline static unsigned int abs(unsigned int x) {
    return x;
  }
};

template <typename T>
inline T absolute_value(const T x) {
  return maths::abs<T>(x);
}

template <typename T>
T doit() {
  T a = -3;
  std::cout << absolute_value<T>(a) << '\n';
}

auto main() -> int
{
  doit<int>();
  return 0;
}
