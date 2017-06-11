#include <cstdlib>
#include <iostream>

template <typename T>
T xalloc(T** p, T N) {
  *p = (T*)malloc(sizeof(T) * N);
  if (p != NULL)
    for (T i = 0; i < N; ++i)
      *(*p + i) = i;
}

template <typename T, T N>
T doit () {
  T* p;
  std::cout << sizeof(p) << '\n';
  xalloc<T>(&p, N);
  std::cout << sizeof(p) << '\n';
  for (T i = 0; i < N; ++i)
    std::cout << *(p + i) << '\n';

  free(p);
  p = NULL;
}

auto main() -> int
{
  doit<int, 10>();
  return 0;
}
