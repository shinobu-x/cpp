#include <iostream>
#include <random>

template <typename T>
void merge_and_sort(T* a1, T* a2, T n) {
}

template <typename T, T N>
void doit() {
  T a[N];
  std::random_device rd;

  for (T i=0; i<N; ++i) {
    a[i] = rd();
    std::cout << a[i] << '\n';
  }
}

auto main() -> int
{
  doit<size_t, 11>();
  return 0;
}
