#include <iostream>

template <typename T>
bool is_product_negative(T x, T y) {
  return x<0 ^ y<0;
}

bool is_product_negative(short x, short y) {
  return int(x)*int(y) < 0;
}

bool is_product_negative(unsigned int x, unsigned int y) {
  return false;
}

bool is_product_negative(unsigned long x, unsigned long y) {
  return false;
}

template <typename T>
T doit() {
}

auto main() -> int 
{
  doit<int>();
  return 0;
}
