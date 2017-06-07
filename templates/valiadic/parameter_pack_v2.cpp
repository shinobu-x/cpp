#include <iostream>
#include <type_traits>

template <typename T, typename... A>
struct count; /// Nothing

template <typename T, typename... A>
struct count<T, T, A...> {
  static const int value = 1 + count<T, A...>::value;
};

template <typename T, typename T2, typename... A>
struct count<T, T2, A...> : count<T, A...> {}; /// Empty

template <typename T>
struct count<T> : std::integral_constant<int, 0> {}; /// Empty

template <typename T>
struct doit {
  template <typename U, typename... A>
  int assert() {
    static_assert(count<U, A...>::value <= 1, "Error");
  }

  template <typename... N>
  void expand_all(N...) {}

  template <typename... A>
  void no_duplicates(A... a) {
    expand_all(assert<A, A...>()...);
  }
};

auto main() -> int
{
  doit<void> a;
  std::cout << a.assert<int, double>() << '\n';
  return 0;
}
