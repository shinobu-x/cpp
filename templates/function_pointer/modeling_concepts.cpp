#include <iostream>

template <typename T1, typename T2>
struct static_assert_can_copy_T1_to_T2 {
  static void concept_check(T1 x, T2 y) {
    T2 z(x);
    y = x;
  }

  static_assert_can_copy_T1_to_T2() {
    void (*f)(T1, T2) = concept_check;
  }
};

template <typename T>
T sqrt(T x) {
  static_assert_can_copy_T1_to_T2<T, double> CHECK1;

  T a = 2;
  double b = 3.14;

  CHECK1.concept_check(a, b);

  std::cout << a << '\n';
  std::cout << b << '\n';
}

template <typename T>
class math_operations : static_assert_can_copy_T1_to_T2<T, double> {};

template <typename T>
T doit() {
  math_operations<T> a;
  sqrt<T>(2);
}

auto main() -> int
{
  doit<int>();
  return 0;
}
