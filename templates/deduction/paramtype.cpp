#include <iostream>

template <typename T>
void f(T a){
  std::cout << a << '\n';
} 

auto main() -> int
{
  int a = 1;
  const int b = 2;
  const int &c = 3;
  const int *d = &c;
  const char* const e = "a";

  f(a);   /// paramtype is int
  f(b);   /// paramtype is const int
  f(c);   /// paramtype is const int&
  f(*d);  /// paramtype is int*
  f(4);   /// paramtype is int&&
  f(*e);  /// paramtype const char* const

  return 0;
}
