#include <memory>
#include <utility>

#include <cassert>

struct A {
  mutable unsigned int hash;
  A() : hash(0) {}

  int f0() {
    f1(17); return 0; }
  int g0() const {
    g1(17); return 0; }

  int f1(int a1) {
    hash = (hash * 17041 + a1) % 32768; return 0; }
  int g1(int a1) const {
    hash = (hash * 17041 + a1 * 2) % 32768; return 0; }

  int f2(int a1, int a2)
  { f1(a1); f1(a2); return 0; }
  int g2(int a1, int a2) const
  { g1(a1); g1(a2); return 0; }

  int f3(int a1, int a2, int a3)
  { f2(a1, a2); f1(a3); return 0; }
  int g3(int a1, int a2, int a3) const
  { g2(a1, a2); g1(a3); return 0; }

  int f4(int a1, int a2, int a3, int a4)
  { f3(a1, a2, a3); f1(a4); return 0; }
  int g4(int a1, int a2, int a3, int a4) const 
  { g3(a1, a2, a3); g1(a4); return 0; }

  int f5(int a1, int a2, int a3, int a4, int a5)
  { f4(a1, a2, a3, a4); f1(a5); return 0; }
  int g5(int a1, int a2, int a3, int a4, int a5)
  { g4(a1, a2, a3, a4); g1(a5); return 0; }

  int f6(int a1, int a2, int a3, int a4, int a5, int a6)
  { f5(a1, a2, a3, a4, a5); f1(a6); return 0; }
  int g6(int a1, int a2, int a3, int a4, int a5, int a6)
  { g5(a1, a2, a3, a4, a5); g1(a6); return 0; }

  int f7(int a1, int a2, int a3, int a4 ,int a5, int a6, int a7)
  { f6(a1, a2, a3, a4, a5, a6); f1(a7); return 0; }
  int g7(int a1, int a2, int a3, int a4, int a5, int a6, int a7)
  { g6(a1, a2, a3, a4, a5, a6); g1(a7); return 0; }

  int f8(int a1, int a2, int a3, int a4, int a5, int a6, int a7, int a8)
  { f7(a1, a2, a3, a4, a5, a6, a7); f1(a8); return 0; }
  int g8(int a1, int a2, int a3, int a4, int a5, int a6, int a7, int a8)
  { g7(a1, a2, a3, a4, a5, a6, a7); g1(a8); return 0; }

  int f9(int a1, int a2, int a3, int a4, int a5, int a6, int a7, int a8, int a9)
  { f8(a1, a2, a3, a4, a5, a6, a7, a8); f1(a9); return 0; }
  int g9(int a1, int a2, int a3, int a4, int a5, int a6, int a7, int a8, int a9)
  { g8(a1, a2, a3, a4, a5, a6, a7, a8); g1(a9); return 0; }
};

auto main() -> decltype(0) {
  A a;
  A const& ra = a;
  A const* pa = &a;
  std::shared_ptr<A> spa(new A);

  std::mem_fn(&A::f0)(a);
  std::mem_fn(&A::f0)(&a);
  std::mem_fn(&A::f0)(spa);

  std::mem_fn(&A::g0)(a);
  std::mem_fn(&A::g0)(ra);
  std::mem_fn(&A::g0)(&a);
  std::mem_fn(&A::g0)(pa);
  std::mem_fn(&A::g0)(spa);

  std::mem_fn(&A::f1)(a, 1);
  std::mem_fn(&A::f1)(&a, 1);
  std::mem_fn(&A::f1)(spa, 1);

  std::mem_fn(&A::g1)(a, 1);
  std::mem_fn(&A::g1)(ra, 1);
  std::mem_fn(&A::g1)(&a, 1);
  std::mem_fn(&A::g1)(pa, 1);
  std::mem_fn(&A::g1)(spa, 1);

  std::mem_fn(&A::f2)(a, 1, 2);
  std::mem_fn(&A::f2)(&a, 1, 2);
  std::mem_fn(&A::f2)(spa, 1, 2);

  std::mem_fn(&A::g2)(a, 1, 2);
  std::mem_fn(&A::g2)(ra, 1, 2);
  std::mem_fn(&A::g2)(&a, 1, 2);
  std::mem_fn(&A::g2)(pa, 1, 2);
  std::mem_fn(&A::g2)(spa, 1, 2);

  std::mem_fn(&A::f3)(a, 1, 2, 3);
  std::mem_fn(&A::f3)(&a, 1, 2, 3);
  std::mem_fn(&A::f3)(spa, 1, 2, 3);

  std::mem_fn(&A::g3)(a, 1, 2, 3);
  std::mem_fn(&A::g3)(ra, 1, 2, 3);
  std::mem_fn(&A::g3)(&a, 1, 2, 3);
  std::mem_fn(&A::g3)(pa, 1, 2, 3);
  std::mem_fn(&A::g3)(spa, 1, 2, 3);

  std::mem_fn(&A::f4)(a, 1, 2, 3, 4);
  std::mem_fn(&A::f4)(&a, 1, 2, 3, 4);
  std::mem_fn(&A::f4)(spa, 1, 2, 3, 4);

  std::mem_fn(&A::g4)(a, 1, 2, 3, 4);
  std::mem_fn(&A::g4)(ra, 1, 2, 3, 4);
  std::mem_fn(&A::g4)(&a, 1, 2, 3, 4);
  std::mem_fn(&A::g4)(pa, 1, 2, 3, 4);
  std::mem_fn(&A::g4)(spa, 1, 2, 3, 4);

  std::mem_fn(&A::f5)(a, 1, 2, 3, 4, 5);
  std::mem_fn(&A::f5)(&a, 1, 2, 3, 4, 5);
  std::mem_fn(&A::f5)(spa, 1, 2, 3, 4, 5);

  std::mem_fn(&A::g5)(a, 1, 2, 3, 4, 5);
  std::mem_fn(&A::g5)(ra, 1, 2, 3, 4, 5);
  std::mem_fn(&A::g5)(&a, 1, 2, 3, 4, 5);
  std::mem_fn(&A::g5)(pa, 1, 2, 3, 4, 5);
  std::mem_fn(&A::g5)(spa, 1 ,2 ,3 ,4 ,5);

  std::mem_fn(&A::f6)(a, 1, 2, 3, 4, 5, 6);
  std::mem_fn(&A::f6)(&a, 1, 2, 3, 4, 5, 6);
  std::mem_fn(&A::f6)(spa, 1, 2, 3, 4, 5, 6);

  std::mem_fn(&A::g6)(a, 1, 2, 3, 4, 5, 6);
  std::mem_fn(&A::g6)(ra, 1, 2, 3, 4, 5, 6);
  std::mem_fn(&A::g6)(&a, 1, 2, 3, 4, 5, 6);
  std::mem_fn(&A::g6)(pa, 1, 2, 3, 4, 5, 6);
  std::mem_fn(&A::g6)(spa, 1, 2, 3, 4, 5, 6);

  std::mem_fn(&A::f7)(a, 1, 2, 3, 4, 5, 6, 7);
  std::mem_fn(&A::f7)(&a, 1, 2, 3, 4, 5, 6, 7);
  std::mem_fn(&A::f7)(spa, 1, 2, 3, 4, 5, 6, 7);

  std::mem_fn(&A::g7)(a, 1, 2, 3, 4, 5, 6, 7);
  std::mem_fn(&A::g7)(ra, 1, 2, 3, 4, 5, 6, 7);
  std::mem_fn(&A::g7)(&a, 1, 2, 3, 4, 5, 6, 7);
  std::mem_fn(&A::g7)(pa, 1, 2, 3, 4, 5, 6, 7);
  std::mem_fn(&A::g7)(spa, 1, 2, 3, 4, 5, 6, 7);

  std::mem_fn(&A::f8)(a, 1, 2, 3, 4, 5, 6, 7, 8);
  std::mem_fn(&A::f8)(&a, 1, 2, 3, 4, 5, 6, 7, 8);
  std::mem_fn(&A::f8)(spa, 1, 2, 3, 4, 5, 6, 7, 8);

  std::mem_fn(&A::g8)(a, 1, 2, 3, 4, 5, 6, 7, 8);
  std::mem_fn(&A::g8)(ra, 1, 2, 3, 4, 5, 6, 7, 8);
  std::mem_fn(&A::g8)(&a, 1, 2, 3, 4, 5, 6, 7, 8);
  std::mem_fn(&A::g8)(pa, 1, 2, 3, 4, 5, 6, 7, 8);
  std::mem_fn(&A::g8)(spa, 1, 2, 3, 4, 5, 6, 7, 8);

  std::mem_fn(&A::f9)(a, 1, 2, 3, 4, 5, 6, 7, 8, 9);
  std::mem_fn(&A::f9)(&a, 1, 2, 3, 4, 5, 6, 7, 8, 9);
  std::mem_fn(&A::f9)(spa, 1, 2, 3, 4, 5, 6, 7, 8, 9);

  std::mem_fn(&A::g9)(a, 1, 2, 3, 4, 5, 6, 7, 8, 9);
  std::mem_fn(&A::g9)(ra, 1, 2, 3, 4, 5, 6, 7, 8, 9);
  std::mem_fn(&A::g9)(&a, 1, 2, 3, 4, 5, 6, 7, 8, 9);
  std::mem_fn(&A::g9)(pa, 1, 2, 3, 4, 5, 6, 7, 8, 9);
  std::mem_fn(&A::g9)(spa, 1, 2, 3, 4, 5, 6, 7, 8, 9);

  assert(std::mem_fn(&A::hash)(a) == 17610 &&
    std::mem_fn(&A::hash)(sa) == 2155);

  return 0;
}
