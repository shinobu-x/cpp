#include <iostream>

template <typename T>
struct BaseInner {};

template <typename T>
struct Outer {
  template <typename U>
  struct Inner : public BaseInner<U> {
    int value;

    Inner& operator=(const BaseInner<U>& that) {
      static_cast<BaseInner<U>& >(*this) = that;
      return *this;
    }
  };
};

template <typename T>
T doit() {
  Outer<double>::Inner<T> a;
  Outer<int>::Inner<T> b;
  a = b;
}

auto main() -> int
{
  doit<int>();
  return 0;
}
