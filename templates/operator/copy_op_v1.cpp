#include <iostream>

template <typename T>
struct TEST_T {
  const int value = 1;

  TEST_T& operator=(const TEST_T& that) {
    static_cast<T&>(*this) = that;
    return *this;
  }
};

template <typename T>
T doit() {
 TEST_T<T> a;
 return a.value;
}

auto main() -> int
{
  std::cout << doit<int>() << '\n';
  return 0;
}
