#include <climits>
#include <iostream>

template <typename T, T N>
struct static_parameter {}; // Empty

template <typename T, T N>
struct static_value : static_parameter<T, N> {
  const static T value = N;
};

template <size_t X, size_t Y>
struct helper {
  static const size_t v = (X >> (Y/2));
  static const int value =
    (v ? Y/2 : 0) + helper<(v ? v : X), (v ? Y-Y/2 : Y/2)>::value;
};

template <size_t X>
struct helper<X, 1> {
  static const int value = X ? 0 : -1;
};

template <size_t X>
struct static_highest_bit
  : static_value<int, helper<X, CHAR_BIT*sizeof(size_t)>::value> {};

template <typename T>
struct binary_ordering_traits {
  static bool gt(const T& x, const T& y) { return x>y; }
  static bool lt(const T& x, const T& y) { return x<y; }
  static bool ge(const T& x, const T& y) { return x>=y; }
  static bool le(const T& x, const T& y) { return x<=y; }
};

template <typename T>
struct binary_equivalence_traits {
  static bool eq(const T& x, const T& y) { return x==y; }
  static bool ne(const T& x, const T& y) { return x!=y; }
};

template <typename T>
struct binary_ordering_less_traits {
  static bool gt(const T& x, const T& y) { return x>y; }
  static bool lt(const T& x, const T& y) { return x<y; }
  static bool ge(const T& x, const T& y) { return !(x>y); }
  static bool le(const T& x, const T& y) { return !(x<y); }
};

template <typename T>
struct binary_equivalence_equal_traits {
  static bool eq(const T& x, const T& y) { return x==y; }
  static bool ne(const T& x, const T& y) { return !(x==y); }
};

template <typename T>
struct binary_equivalence_less_traits {
  static bool eq(const T& x, const T& y) { return !(x>y) && !(x<y); }
  static bool ne(const T& x, const T& y) { return x>y || x<y; }
};

enum {
  ALL,
  LESS_AND_EQ,
  LESS
};

template <typename T, int = ALL>
struct binary_comparison_traits
  : binary_ordering_traits<T>, binary_equivalence_traits<T> {};

template <typename T>
struct binary_comparison_traits<T, LESS>
  : binary_ordering_less_traits<T>, binary_equivalence_less_traits<T> {};

template <typename T>
struct binary_comparison_traits<T, LESS_AND_EQ>
  : binary_ordering_less_traits<T>, binary_equivalence_less_traits<T> {};

namespace native {
  enum {
    lt = 1,
    le = 2,
    gt = 4,
    ge = 8,
    eq = 16,
    ne = 32
  };
}

template <typename T, int F>
struct more_binary_comparison_traits;

template <typename T>
struct more_binary_comparison_traits<T, native::lt> {
  static bool lt(const T& x, const T& y) { return x<y; }
};

template <typename T>
struct more_binary_comparison_traits<T, native::le> {
  static bool le(const T& x, const T& y) { return x<=y; }
};

template <typename T, unsigned F>
struct and_binary_comparison_traits;  // Nothing

template <typename T>
struct and_binary_comparison_traits<T, 0> {};  // Empty

template <typename T>
struct and_binary_comparison_traits<T, native::lt> {
  static bool lt(const T& x, const T& y) { return x<y; }
};

template <typename T>
struct and_binary_comparison_traits<T, native::gt> {
  static bool gt(const T& x, const T& y) { return x>y; }
};

template <typename T, unsigned F>
struct and_binary_comparison_traits
  : and_binary_comparison_traits<T, F & (1 << static_highest_bit<F>::value)>,
    and_binary_comparison_traits<T, F - (1 << static_highest_bit<F>::value)>
{}; // Too much

template <typename T>
T doit() {
  typedef T type_t;
  typedef and_binary_comparison_traits<
    type_t, native::lt | native::gt> traits_t;
  type_t t1, t2;
  traits_t::lt(t1, t2);
}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
