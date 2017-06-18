#include <iostream>
#include <random>

template <typename T, T N>
void doit() {
  std::random_device rd;
  T* a[N];
  for (T i=0; i<N; ++i) {
    *(a+i) = (T*)(malloc(sizeof(T)));
    **(a+i) = rd();
  }

  T i = 0, j=N-1;
  T k = (i+j)/2;
  T l = k; 
  T m = N-k; 

  T *a1[l], *a2[m];

  for (T i=0; i<l; ++i) {
    *(a1+i) = (T*)(malloc(sizeof(T)));
    **(a1+i) = **(a+i);
  }

  for (T i=0; i<m; ++i) {
    *(a2+i) = (T*)(malloc(sizeof(T)));
    **(a2+i) = **(a+((m-1)+i));
  }

  i=0, j=0, k=0;
  while (i < l && j < m) {
    if (**(a1+i) <= **(a2+j)) {
      **(a+k) = **(a1+i);
      ++i;
    } else {
      **(a+k) = **(a2+j);
      ++j;
    }
    ++k;
  }
}

auto main() -> int
{
  doit<unsigned int, 11>();
  return 0;
}
