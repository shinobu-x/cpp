#include <iostream>

template <typename T>
struct is_pointer_type {
  static const int value = 1;
};

template <>
struct is_pointer_type<void*> {
  static const int value = 2;
};

template <typename T>
struct is_pointer_type<T*> {
  static const int value = 3;
};

template <typename T, typename U>
void f() {
  int a = is_pointer_type<int*>::value;
  int b = is_pointer_type<void*>::value;
  int c = is_pointer_type<int>::value;

  // ******

  std::cout << a << '\n';
  std::cout << b << '\n';
  std::cout << c << '\n';
}

auto main() -> int
{
  f<int, bool>();
  return 0;
}
