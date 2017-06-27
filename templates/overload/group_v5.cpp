#include <cmath>
#include <iostream>

template <bool cond_t>
struct selector {};

template <typename scalar_t>
struct maths {
  static scalar_t abs(const scalar_t& x, selector<false>) {
    return x<0 ? -x : x;
  }

  static scalar_t abs(const scalar_t& x, selector<true>) {
    return x.abs();
  }
};

template <>
struct maths<double> {
  template <bool cond_t>
  static double abs(const double x, selector<cond_t>) {
    return std::fabs(x);
  }
};

template <typename scalar_t>
inline scalar_t absolute_value(const scalar_t& x) {
  typedef selector<false> select_t;
  return maths<scalar_t>::abs(x, select_t());
}

template <typename T>
T doit() {
  T a = -1;
  std::cout << absolute_value<T>(a) << '\n';
}

auto main() -> int
{
  doit<int>();
  return 0;
}
