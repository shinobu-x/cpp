#include <functional>

#include <cassert>

int f(int a) { return a; }

auto main() -> decltype(0) {
  assert(std::bind(f, std::placeholders::_1)(1) == 1);
  assert(std::bind(f, std::placeholders::_2)(1,2) == 2);
  assert(std::bind(f, std::placeholders::_3)(1,2,3) == 3);
  assert(std::bind(f, std::placeholders::_4)(1,2,3,4) == 4);
  assert(std::bind(f, std::placeholders::_5)(1,2,3,4,5) == 5);
  assert(std::bind(f, std::placeholders::_6)(1,2,3,4,5,6) == 6);
  assert(std::bind(f, std::placeholders::_7)(1,2,3,4,5,6,7) == 7);
  assert(std::bind(f, std::placeholders::_8)(1,2,3,4,5,6,7,8) == 8);
  assert(std::bind(f, std::placeholders::_9)(1,2,3,4,5,6,7,8,9) == 9);
}
