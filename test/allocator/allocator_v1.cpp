#include <iostream>

template <typename T>
T xalloc(T** a, T s, T v) {
  *a = (T*)(malloc(sizeof(T)*s));
  if (*a != NULL)
    for (int i = 0; i < s; ++i)
      *(*a + i) = v;
}

template <typename T>
T doit() {
  T* v = NULL;
  T x = 5;
  T y = 1;
  xalloc(&v, x, y);

//  for (T i = 0; i < x; ++i)
//    std::cout << v[i] << '\n';

  v = NULL;
  free(v);
}

auto main() -> int
{
  typedef int type;
  type* v = NULL;
  doit<type>();
} 
