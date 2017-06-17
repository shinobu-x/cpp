#include <iostream>

template <typename T, T N>
void doit() {
  T* a[N];
  for (T i = 0; i < N; ++i) {
    *(a + i) = (T*)(malloc(sizeof(T)));
    **(a + i) = i;
  }
}

auto main() -> int
{
  doit<int, 10>();
  return 0;
}
