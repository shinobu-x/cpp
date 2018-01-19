#include <iostream>
#include <random>

template <typename T, T N>
T doit() {
  T* a[N];
  std::random_device rd;
  for (T i=0; i<N; ++i) {
    *(a+i) = (T*)(malloc(sizeof(T)));
    **(a+i) = rd();
  }

  T t;

  for (T i=0; i<N-1; ++i) {
    if (**(a+i) > **(a+(i+1))) {
      t = **(a+i);
      **(a+i) = **(a+(i+1));
      **(a+(i+1)) = t;
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
  
