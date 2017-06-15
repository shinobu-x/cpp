#include <iostream>

template <typename T>
struct type1_t {
  const static T value = 1;
};

template <typename T>
struct type2_t {
  const static T value = 2;
};

template <typename T1, typename T2>
struct naive_or {
  static const bool condition = (T1::value || T2::value);
};

template <typename T1, typename T2>
void doit() {
  type1_t<T1> t1;
  type2_t<T2> t2;
  naive_or<type1_t<T1>, type2_t<T2> > n;
  if (n.condition)
    std::cout << "True" << '\n';
  else
    std::cout << "False" << '\n';
}

auto main() -> int {
  doit<int, int>();
  return 0;
}
