#include <iostream>

template <typename T>
class Ptr {
public:
  T* operator->() const noexcept {
    return p_;
  }
private:
  T* p_;
};

template <typename T>
T doit() {
  Ptr<T> p;
  T a = 10;
  T b = 20;
  p = &a;
  std::cout << p << '\n';
}
auto main() -> int
{
  doit<int>();
  return 0;
}
