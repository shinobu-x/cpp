#include <memory>
#include <utility>

#include <cassert>

struct A {
  static int count;
  A() { ++count; }
  A(A const &that) {}
  virtual ~A() { --count; }
};

int A::count = 0;

struct B : A {
  static int count;
  B() : A() { ++count; }
  B(B const &that) : A(that) { ++count; }
  virtual ~B() { --count; }
};

int B::count = 0;

void reset() {
  A::count = 0;
  B::count = 0;
}

auto main() -> decltype(0) {
  reset();
  assert(A::count == 0 && B::count == 0);
  {
    std::unique_ptr<B> s(new B);
    A* p = s.get();
    std::unique_ptr<A> s2(new A);
    assert(A::count == 2);
    assert(B::count == 1);
    s2 = std::move(s);
    assert(s2.get() == p);
    assert(s.get() == 0);
    assert(A::count == 1);
    assert(B::count == 1);
  }
  assert(A::count == 0 && B::count == 0);
}
