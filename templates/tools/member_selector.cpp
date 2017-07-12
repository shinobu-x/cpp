#include <climits>
#include <iostream>

template <typename T, T N>
struct static_parameter {};  // Empty

template <typename T, T N>
struct static_value : static_parameter<T, N> {
  const static T value = N;
};

template <size_t X, size_t Y>
struct helper {
  static const size_t v = (X >> (X/2));
  static const size_t value =
    (v ? Y/2 : 0) + helper<(v ? v : X), (v ? Y-Y/2 : Y/2)>::value;
};

template <size_t X>
struct helper<X, 1> {
  static const int value = X ? 0 : -1;
};

template <size_t X>
struct static_highest_bit
  : static_value<int, helper<X, CHAR_BIT*sizeof(size_t)>::value> {};

enum {
  empty = 0,
  year = 1,
  month = 2,
  day = 4,
};

template <unsigned CODE>
struct time_val;

template <>
struct time_val<empty> {};  // Empty!

template <>
struct time_val<year>
{ int year; };

template <>
struct time_val<month>
{ short month; };

template <>
struct time_val<day>
{ int day; };

template <unsigned CODE>
struct time_val
  : public time_val<CODE & static_highest_bit<CODE>::value>
  , public time_val<CODE - static_highest_bit<CODE>::value> {};

template <unsigned CODE>
time_val<year> get(const time_val<CODE>& t) {
  time_val<year> r;
  return r;
}

template <typename T>
void doit() {
}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
