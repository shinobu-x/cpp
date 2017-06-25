#include <iostream>
#include <vector>

template <typename T>
struct F {
  T f1(T);
  T f2(T*);
  T* f3(T);
  T* f4(T*);
  T (*f5)(T);
  T (*f6)(T*);
  T* (*f7)(T);
  T* (*f8)(T*);
};

template <typename T>
T g0(T* a) {
  return *a = (*a)*(*a);
}

int (*f0)(int*);

template <typename T>
T doit() {
  F<T> f;
  T a = 3;
  f0 = g0<T>;
  f.f6 = g0<T>;
  std::cout << f0(&a) << '\n';
  std::cout << f.f6(&a) << '\n';
}

auto main() -> int
{
  doit<int>();
  return 0;
}
