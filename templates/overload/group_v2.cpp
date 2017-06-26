#include <iostream>

template <bool C>
struct selector {};

template <typename T>
bool is_product_negative(T x, T y, selector<false>) {
  return x<0 ^ y<0;
}

template <typename T>
bool is_product_negative(T x, T y, selector<true>) {
  return int(x)*int(y) < 0;
}

template <typename T>
bool is_product_negative(T x, T y) {
  typedef selector<(sizeof(T) < sizeof(int))> small_int_t;
  return is_product_negative(x, y, small_int_t());
}


template <typename T>
T doit() {
}

auto main() -> int 
{
  doit<int>();
  return 0;
}
