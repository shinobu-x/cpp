#include <cstdlib>
#include <iostream>

template <typename T, T N>
T doit() {
  T* a = (T*)(malloc(N*sizeof(T)));
  T q = N;
  T* p = &q;
  T n = 0;

  for (T i = 0; i < N; ++i) 
    a[i] = i;

  for (T j = 0; j < N; ++j) {
    /**
     * Access to memory
     *  #1 Read: i
     *  #2 Read: a[i]
     *  #3 Read: p
     *  #4 Read: Data pointed to by p
     *  #5 Write: n
     */ 
    n = *p + a[j];
    // Read: n
    std::cout << n << '\n';
  }

  free(a);
  a = NULL;
}

auto main() -> int
{
  doit<int, 10>();
  return 0;
}
