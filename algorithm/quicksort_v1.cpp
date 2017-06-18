#include <iostream>
#include <random>

template <typename T, T N>
T doit() {
  std::random_device rd;
  T* a[N];
  for (T i=0; i<N; ++i) {
    *(a+i) = (T*)(malloc(sizeof(T)));
    **(a+i) = rd();
  }

  T i=0, j=N-1;

  T x = **(a+((i+j)/2));

  do {
    while (**(a+i) < x)
      ++i;
    while (x < **(a+j))
      --j;

    if (i <= j) {
      if (i < j) {
        T t = **(a+i);
        **(a+i) = **(a+j);
        **(a+j) = t;
      }
      ++i;
      --j;
    }
  } while(i <= j);

  for (T i=0; i<N; ++i)
    std::cout << **(a+i) << '\n';
}

auto main() -> int
{
  doit<unsigned int, 10>();
  return 0;
}
