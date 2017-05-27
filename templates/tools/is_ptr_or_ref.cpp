#include <iostream>

template <typename T>
struct is_ptr_or_ref;

template <typename T>
struct is_ptr_or_ref<const T&> {
  const static int value = 1;
};

template <typename T>
struct is_ptr_or_ref<T*> {
  const static int value = 2;
};

template <typename T>
void f() {
  int a = is_ptr_or_ref<T*>::value;
  int b = is_ptr_or_ref<const T&>::value;

  // ******
 
  std::cout << a << '\n';
  std::cout << b << '\n';
}

auto main() -> int
{
  f<int>();
  return 0;
}
