#include <iostream>

template <typename T>
T xfree(T** p) {
  if (p != NULL && *p != NULL) {
    free(*p);
    *p = NULL;
  }
}

template <typename T>
T xalloc(T** p, T s, T v) {
  *p = (T*)(malloc(sizeof(T)*s));

  if (*p != NULL)
    for (T i = 0; i < s; ++i)
      *(*p + i) = v;
}

template <typename T>
T doit() {
  T* v = NULL;
  T a = 5, b = 1;
  xalloc<int>(&v, a, b);
  for (int i = 0; i < a; ++i)
    std::cout << v[i] << '\n';

  xfree<void>((void**)&v);
}

auto main() -> int
{
  doit<int>();
  return 0;
}
