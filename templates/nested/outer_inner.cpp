#include <iostream>

template <typename T>
class Outer {
public:
  template <typename U>
  class Inner {
  public:
    const static U value = 1;
  };
};

template <typename T>
T doit() {
  T v = Outer<double>::Inner<T>::value;
  return v;
}

auto main() -> int
{
  std::cout << doit<int>() << '\n';
  return 0;
}
