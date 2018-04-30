#include <iostream>
#include "../HPP/InitData.hpp"

auto main() -> decltype(0) {
  float* a;
  int s = 100;
  a = (float*)malloc(sizeof(float)*s);
  InitData(a, s);

  for (int i = 0; i < s; ++i) {
    std::cout << a[i] << '\n';
  }

  return 0;
}
