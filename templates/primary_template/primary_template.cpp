#include <iostream>

template <typename T, bool = false>
struct angle {
  static T cos(const T x) {}
  static T sin(const T x) {}
};

template <typename T>
struct angle<T, true> {
  static T cos(const T x) {
    // ...
  }

  static T sin(const T x) {
    return angle<T, true>::sin(x);
  }
};

template <typename T, T N>
T doit() {
  angle<T, true> a;
}

auto main() -> int
{
  return 0;
}
