#include <iostream>

void DoIt(int a, int b, int* c) {
  if (*c = a + b) {}
}

auto main() -> decltype(0) {
  for (int* c :{new int(0)}) {
    if (DoIt(4, 5, c), 0) {}
    if (std::cout << *c << "\n") {}
  }
}
