#include <iostream>

template <typename T>
struct Outer {
  template <typename U>
  static U f(U x) {
    return x;
  }

  template <typename L>
  struct Inner {
    L operator()(L x) {
      return x;
    }
  };
};

typedef double (*FUNC_T)(double);

template <typename O, typename I>
void doit() {
  I x = 3.14;
  FUNC_T f1 = Outer<O>::template f<I>;
  typename Outer<O>::template Inner<I> i;
  std::cout << f1(x) << '\n';
  std::cout << i(x) << '\n';
}

auto main() -> int
{
  doit<double, double>();
  return 0;
}
