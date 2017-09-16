#include <boost/bind.hpp>

#include <cassert>

int a, b, c, d, e, f, g, h, i;

void reset() {
  a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0, h = 0, i = 0;
}

void f1(
  int& a) {
  a = 1;
}

void f2(
  int& a, int& b) {
  a = 1, b = 2;
}

void f3(
  int& a, int& b, int& c) {
  a = 1, b = 2, c = 3;
}

void f4(
  int& a, int& b, int& c, int& d) {
  a = 1, b = 2, c = 3, d = 4;
}

void f5(
  int& a, int& b, int& c, int& d, int& e) {
  a = 1, b = 2, c = 3, d = 4, e = 5;
}

void f6(
  int& a, int& b, int& c, int& d, int& e, int& f) {
  a = 1, b = 2, c = 3, d = 4, e = 5, f = 6;
}

void f7(
  int& a, int& b, int& c, int& d, int& e, int& f, int& g) {
  a = 1, b = 2, c = 3, d = 4, e = 5, f = 6, g = 7;
}

void f8(
  int& a, int& b, int& c, int& d, int& e, int& f, int& g, int& h) {
  a = 1, b = 2, c = 3, d = 4, e = 5, f = 6, g = 7, h = 8;
}

void f9(
  int& a, int& b, int& c, int& d, int& e, int& f, int& g, int& h, int& i) {
  a = 1, b = 2, c = 3, d = 4, e = 5, f = 6, g = 7, h = 8, i = 9;
}

auto main() -> decltype(0) {
  {
    reset();
    assert(a == b == c == d == e == f == g == h == i == 0);
    boost::bind(f1,
      _1)(a);
    assert(a ==1);
  }
  {
    reset();
    assert(a == b == c == d == e == f == g == h == i == 0);
    boost::bind(f2,
      _1, _2)(a, b);
    assert(a == 1); assert(b == 2);
  }
  {
    reset();
    assert(a == b == c == d == e == f == g == h == i == 0);
    boost::bind(f3,
      _1, _2, _3)(a, b, c);
    assert(a == 1); assert(b == 2); assert(c == 3);
  }
  {
    reset();
    assert(a == b == c == d == e == f == g == h == i == 0);
    boost::bind(f4,
      _1, _2, _3, _4)(a, b, c, d);
    assert(a == 1); assert(b == 2); assert(c == 3); assert(d == 4);
  }
  {
    reset();
    assert(a == b == c == d == e == f == g == h == i == 0);
    boost::bind(f5,
      _1, _2, _3, _4, _5)(a, b, c, d, e);
    assert(a == 1); assert(b == 2); assert(c == 3); assert(d == 4);
    assert(e == 5);
  }
  {
    reset();
    assert(a == b == c == d == e == f == g == h == i == 0);
    boost::bind(f6,
      _1, _2, _3, _4, _5, _6)(a, b, c, d, e, f);
    assert(a == 1); assert(b == 2); assert(c == 3); assert(d == 4);
    assert(e == 5); assert(f == 6);
  }
  {
    reset();
    assert(a == b == c == d == e == f == g == h == i == 0);
    boost::bind(f7,
      _1, _2, _3, _4, _5, _6, _7)(a, b, c, d, e, f, g);
    assert(a == 1); assert(b == 2); assert(c == 3); assert(d == 4);
    assert(e == 5); assert(f == 6); assert(g == 7);
  }
  {
    reset();
    assert(a == b == c == d == e == f == g == h == i == 0);
    boost::bind(f8,
      _1, _2, _3, _4, _5, _6, _7, _8)(a, b, c, d, e, f, g, h);
    assert(a == 1); assert(b == 2); assert(c == 3); assert(d == 4);
    assert(e == 5); assert(f == 6); assert(g == 7); assert(h == 8);
  }
  {
    reset();
    assert(a == b == c == d == e == f == g == h == i == 0);
    boost::bind(f9,
      _1, _2, _3, _4, _5, _6, _7, _8, _9)(a, b, c, d, e, f, g, h, i);
    assert(a == 1); assert(b == 2); assert(c == 3); assert(d == 4);
    assert(e == 5); assert(f == 6); assert(g == 7); assert(h == 8);
    assert(i == 9);
  }

  return 0;
}
