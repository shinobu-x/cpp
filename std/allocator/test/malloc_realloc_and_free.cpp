#include <cstdlib>
#include <iostream>

template <typename T, T N>
T doit() {
  T *p = (int*)(malloc((sizeof(T)*N)));

  if (p == NULL)
    return 1;

  for (int i = 0; i < N; ++i) {
    p[i] = i;
    std::cout << &p[i] << ": " << p[i] << '\n';
  }

  p = (int*)(realloc(p, (N + N) * sizeof(T)));

  if (p == NULL)
    return 1;

  for (int i = 0; i < (N + N); ++i) {
    p[i] = i;
    std::cout << &p[i] << ": " << p[i] << '\n';
  }

  free(p);

  p = NULL;

  if (p == NULL)
    std::cout << "NULL" << '\n';
  else
    std::cout << "Non NULL" << '\n';
}

auto main() -> int
{
  doit<int, 10>();
  return 0;
}
