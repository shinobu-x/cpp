#include <iostream>
#include <random>

template <typename T, T N>
T doit() {
  std::random_device rd;
  T *a[N];
  for (T i=0; i<N; ++i) {
    *(a+i) = (T*)(malloc(sizeof(T)));
    **(a+i) = rd();
  }

  T t;
  /**
   * N=10; j=9;
   * i=0; 9,8,7,..,1>0;
   * i=1; 9,8,7,..,2>1;
   *  ...
   * i=8; 9>8
   */
  for (T i=0; i<N-1; ++i) {
    for (T j=N-1; j>i; --j) {
      if (**(a+j) < **(a+(j-1))) {
        T t = **(a+j);
        **(a+j) = **(a+(j-1));
        **(a+(j-1)) = t;
      }
    }
  }

  for (T i=0; i<N; ++i)
    std::cout << **(a+i) << '\n';     
}

auto main() -> int
{
  doit<unsigned int, 10>();
  return 0;
} 
