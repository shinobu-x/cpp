#include <cmath>
#include <iostream>

#define M_F(NAME, VALUE)            \
static T NAME() {                   \
  static const T NAME##_ = (VALUE); \
  return NAME##_;                   \
}

template <typename T>
struct TEST_T {
  M_F(pi1, std::acos(T(-1)));
  M_F(pi2, 2*std::acos(T(-1)));
  M_F(pi3, std::acos(T(0)));
  M_F(pi4, std::atan(T(1)));
  M_F(log2, std:: log(T(2)));
};

template <typename T>
T doit() {
  T x = TEST_T<T>::pi2();
/**
 *  static T pi2() {
 *    static const pi2_ = VALUE;
 *    return pi2_;
 *  }
 */
  return x;
}
auto main() -> int
{
  std::cout << doit<double>() << '\n';
  return 0;
}
