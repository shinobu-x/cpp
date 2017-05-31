#include <algorithm>
#include <cstddef>
#include <iostream>
#include <utility>

#define NS_BEGIN(X) namespace X {
#define NS_END      }

NS_BEGIN(swap_impl)
  template <class T>
  void swap_impl(T& l, T& r) {
    using namespace std;
    swap(l, r);
  }

  template <class T, std::size_t N>
  void swap_impl(T (& l)[N], T (& r)[N]) {
    for (std::size_t i = 0; i < N; ++i)
      ::swap_impl::swap_impl(l[i], r[i]);
  }
NS_END

NS_BEGIN(nonstd)
  template <class T1, class T2>
  void swap(T1& l, T2& r) {
    ::swap_impl::swap_impl(l, r);
  }
NS_END

template <typename T>
T doit() {
  T a = 1, b = 1000;
  std::cout << "a = " << a << " b = " << b << '\n';
  nonstd::swap(a, b);
  std::cout << "a = " << a << " b = " << b << '\n'; 
}

auto main() -> int
{
  doit<int>();
  return 0;
}
