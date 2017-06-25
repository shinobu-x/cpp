#include <iostream>
#include <random>

template <typename T>
void merge_sort(T a[], int l, int r) {
//  std::cout << l << " " << r << '\n';
  if (l >= r) return;

  T t[10];

  T m = (l + r)/2;

  merge_sort<T>(a, l, m);
  merge_sort<T>(a, m + 1, r);

  for (T i=l; i<=m; ++i) 
    t[i] = a[i];

  for (T i=m+1, j=r; i<=r; ++i, --j)
    t[i] = a[i];

  T i=l, j=r;

  for (T k=l; k<=r; ++k)
    (t[i] <= t[j]) ? a[k] = t[++i] : a[k] = t[--j];
}

template <typename T, T N>
T doit() {
  T* a = (T*)(malloc(sizeof(T)*N));
  std::random_device rd;

  for (T i=0; i<N; ++i)
    a[i] = rd();

  for (T i=0; i<N; ++i)
    std::cout << a[i] << '\n';
  std::cout << "\n\n\n\n";

  merge_sort<T>(a, 0, N-1);

  for (T i=0; i<N; ++i)
    std::cout << a[i] << '\n';
}

auto main() -> int
{
  doit<int, 10>();
  return 0;
}
