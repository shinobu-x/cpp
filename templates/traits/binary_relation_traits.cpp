#include <iostream>

template <typename T>
struct binary_relation_traits {
  static bool gt(const T& x, const T& y) { return x>y; }
  static bool lt(const T& x, const T& y) { return x<y; }
  static bool ge(const T& x, const T& y) { return x>=y; }
  static bool le(const T& x, const T& y) { return x<=y; }
  static bool eq(const T& x, const T& y) { return x==y; }
  static bool ne(const T& x, const T& y) { return x!=y; }
};

template <typename T>
T doit() {
  const T a=1, b=2, c=1, d=3;
  binary_relation_traits<T> bt;

  std::cout << a << " > " << b << " ? ";
  if (bt.gt(a, b)) std::cout << "True" << '\n';
  else std::cout << "False" << '\n';
}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
