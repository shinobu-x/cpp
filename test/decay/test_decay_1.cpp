#include <type_traits>

template <typename T>
T doit(T t) {
  if constexpr (std::is_same<T, int>{}) {
    static_assert(std::is_same<T, int*>::value, "ng");
  } else if constexpr (std::is_same<T, int(*)(double)>{}) {
    static_assert(std::is_same<T, int(*)(double)>::value, "ng");
  }
  return t;
}

int f(double) { return 0; }

auto main() -> decltype(0) {
  int a[3] = {1, 2, 3};
  static_assert(std::is_same<std::decay<decltype(a)>::type, int*>::value, "");
  auto x = doit(a);

  static_assert(std::is_same<decltype(f), int(double)>::value, "");
  static_assert(!std::is_same<decltype(f), int(*)(double)>::value, "");

  static_assert(std::is_same<std::decay<int(double)>::type,
    int(*)(double)>::value, "");

  int(*y)(double) = doit(f);
}
