#include <iostream>

template <bool C>
struct selector {};

template <typename T>
struct is_product_negative_t;

template <typename T>
bool is_product_negative(T x, T y) {
  return is_product_negative_t<T>::doit(x, y);
}

template <typename T>
struct is_product_negative_t {
  static bool doit(T x, T y) {
    return x<0 ^ y<0;
  }

  static bool doit(short x, short y) {
    return int(x)*int(y) < 0;
  }

  static bool doit(unsigned, unsigned) {
    return false;
  }
};

template <typename T>
T doit() {
  T a = 1, b = 2;
  std::cout << is_product_negative<T>(a, b) << '\n';
}

auto main() -> int
{
  doit<int>();
  return 0;
}


