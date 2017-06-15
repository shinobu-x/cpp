#include <iostream>

template <bool condition>
struct selector {};

template <bool condition, typename T2>
struct static_or_helper;

template <typename T2>
struct static_or_helper<false, T2> : selector<T2::value> {};

template <typename T2>
struct static_or_helper<true, T2> : selector<true> {};

template <typename T1, typename T2>
struct static_or : static_or_helper<T1::value, T2> {
  void get_value () {
    if (T1::value) std::cout << T1::value << '\n';
    else if (T2::value) std::cout << T1::value << '\n';
  }
};

template <typename T, T N>
struct type1_t {
  const static T value = N;
};

template <typename T, T N>
struct type2_t {
  const static T value = N;
};

template <typename T>
T doit() {
  static_or<type1_t<T, 1>, type2_t<T, 2> > s;
  s.get_value();
}

auto main() -> int
{
  doit<int>();
  return 0;
}
