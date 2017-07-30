#include <functional>
#include <iostream>
#include <memory>

#include <cassert>

void f1(std::unique_ptr<int> p1) {
  assert(*p1 == 1);
}

void f2(std::unique_ptr<int> p1, std::unique_ptr<int> p2) {
  assert(*p1 == 1); assert(*p2 == 2);
}

void f3(std::unique_ptr<int> p1, std::unique_ptr<int> p2, 
  std::unique_ptr<int> p3) {
  assert(*p1 == 1); assert(*p2 == 2); assert(*p3 == 3);
}

void f4(std::unique_ptr<int> p1, std::unique_ptr<int> p2,
  std::unique_ptr<int> p3, std::unique_ptr<int> p4) {
  assert(*p1 == 1); assert(*p2 == 2); assert(*p3 == 3); assert(*p4 == 4);
}

void f5(std::unique_ptr<int> p1, std::unique_ptr<int> p2,
  std::unique_ptr<int> p3, std::unique_ptr<int> p4, std::unique_ptr<int> p5) {
  assert(*p1 == 1); assert(*p2 == 2); assert(*p3 == 3); assert(*p4 == 4);
  assert(*p5 == 5);
}

void f6(std::unique_ptr<int> p1, std::unique_ptr<int> p2,
  std::unique_ptr<int> p3, std::unique_ptr<int> p4, std::unique_ptr<int> p5,
  std::unique_ptr<int> p6) {
  assert(*p1 == 1); assert(*p2 == 2); assert(*p3 == 3); assert(*p4 == 4);
  assert(*p5 == 5); assert(*p6 == 6);
}

void f7(std::unique_ptr<int> p1, std::unique_ptr<int> p2,
  std::unique_ptr<int> p3, std::unique_ptr<int> p4, std::unique_ptr<int> p5,
  std::unique_ptr<int> p6, std::unique_ptr<int> p7) {
  assert(*p1 == 1); assert(*p2 == 2); assert(*p3 == 3); assert(*p4 == 4);
  assert(*p5 == 5); assert(*p6 == 6); assert(*p7 == 7);
}

void f8(std::unique_ptr<int> p1, std::unique_ptr<int> p2,
  std::unique_ptr<int> p3, std::unique_ptr<int> p4, std::unique_ptr<int> p5,
  std::unique_ptr<int> p6, std::unique_ptr<int> p7, std::unique_ptr<int> p8) {
  assert(*p1 == 1); assert(*p2 == 2); assert(*p3 == 3); assert(*p4 == 4);
  assert(*p5 == 5); assert(*p6 == 6); assert(*p7 == 7); assert(*p8 == 8);
}

auto main() -> decltype(0) {
  {
    std::unique_ptr<int> p1(new int(1));
    std::bind(f1, std::placeholders::_1)(
      std::move(p1));
  }
  {
    std::unique_ptr<int> p1(new int(1));
    std::unique_ptr<int> p2(new int(2));
    std::bind(f2, std::placeholders::_1, std::placeholders::_2)(
      std::move(p1), std::move(p2));
  }
  {
    std::unique_ptr<int> p1(new int(1));
    std::unique_ptr<int> p2(new int(2));
    std::unique_ptr<int> p3(new int(3));
    std::bind(f3, std::placeholders::_1, std::placeholders::_2,
      std::placeholders::_3)(
      std::move(p1), std::move(p2), std::move(p3));
  }
  {
    std::unique_ptr<int> p1(new int(1));
    std::unique_ptr<int> p2(new int(2));
    std::unique_ptr<int> p3(new int(3));
    std::unique_ptr<int> p4(new int(4));
    std::bind(f4, std::placeholders::_1, std::placeholders::_2,
      std::placeholders::_3, std::placeholders::_4)(
      std::move(p1), std::move(p2), std::move(p3), std::move(p4));
  }
  {
    std::unique_ptr<int> p1(new int(1));
    std::unique_ptr<int> p2(new int(2));
    std::unique_ptr<int> p3(new int(3));
    std::unique_ptr<int> p4(new int(4));
    std::unique_ptr<int> p5(new int(5));
    std::bind(f5, std::placeholders::_1, std::placeholders::_2,
      std::placeholders::_3, std::placeholders::_4, std::placeholders::_5)(
      std::move(p1), std::move(p2), std::move(p3), std::move(p4),
      std::move(p5));
  }
  {
    std::unique_ptr<int> p1(new int(1));
    std::unique_ptr<int> p2(new int(2));
    std::unique_ptr<int> p3(new int(3));
    std::unique_ptr<int> p4(new int(4));
    std::unique_ptr<int> p5(new int(5));
    std::unique_ptr<int> p6(new int(6));
    std::bind(f6, std::placeholders::_1, std::placeholders::_2,
      std::placeholders::_3, std::placeholders::_4,
      std::placeholders::_5, std::placeholders::_6)(
      std::move(p1), std::move(p2), std::move(p3), std::move(p4),
      std::move(p5), std::move(p6));
  }
  {
    std::unique_ptr<int> p1(new int(1));
    std::unique_ptr<int> p2(new int(2));
    std::unique_ptr<int> p3(new int(3));
    std::unique_ptr<int> p4(new int(4));
    std::unique_ptr<int> p5(new int(5));
    std::unique_ptr<int> p6(new int(6));
    std::unique_ptr<int> p7(new int(7));
    std::bind(f7, std::placeholders::_1, std::placeholders::_2,
      std::placeholders::_3, std::placeholders::_4,
      std::placeholders::_5, std::placeholders::_6,
      std::placeholders::_7)(
      std::move(p1), std::move(p2), std::move(p3), std::move(p4),
      std::move(p5), std::move(p6), std::move(p7));
  }
  {
    std::unique_ptr<int> p1(new int(1));
    std::unique_ptr<int> p2(new int(2));
    std::unique_ptr<int> p3(new int(3));
    std::unique_ptr<int> p4(new int(4));
    std::unique_ptr<int> p5(new int(5));
    std::unique_ptr<int> p6(new int(6));
    std::unique_ptr<int> p7(new int(7));
    std::unique_ptr<int> p8(new int(8));
    std::bind(f8, std::placeholders::_1, std::placeholders::_2,
      std::placeholders::_3, std::placeholders::_4,
      std::placeholders::_5, std::placeholders::_6,
      std::placeholders::_7, std::placeholders::_8)(
      std::move(p1), std::move(p2), std::move(p3), std::move(p4),
      std::move(p5), std::move(p6), std::move(p7), std::move(p8));
  }
}
