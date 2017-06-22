#include <iostream>

template <typename T>
struct add_ptr {
  typedef T* ptr;
};

template <typename T>
T doit() {
  T a = 2;
  typedef typename add_ptr<T>::ptr ptr_t;
  ptr_t b = &a;
  std::cout << *b << '\n'; 
}
auto main() -> int {
  doit<int>();
  return 0;
}
