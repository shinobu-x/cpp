#include <iostream>

template <typename T, int N>
struct F {
  typedef T* pointer_type;
  typedef T& reference_type;

  static const size_t value = sizeof(T)*N;
};

template <typename T, int N>
T doit() {
  //F<T, N> a;

  typedef typename F<T, N>::pointer_type ptr_t;
  typedef typename F<T, N>::reference_type ref_t;

  T t = 10;
  ref_t r = t;
  ptr_t p = &r;

  std::cout << t << '\n';
  std::cout << r << '\n';
  std::cout << *p << '\n';
}

auto main() -> int
{
  doit<int, 5>();
  return 0;
}
