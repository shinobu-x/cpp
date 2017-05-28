#include <iostream>

template <typename T>
struct BASE_T {};

template <typename T>
struct TEST_T : public BASE_T<T> {
  int value;

  TEST_T& operator=(const BASE_T<T>& that) {
    static_cast<BASE_T<T>& >(*this) = that;
    return *this;
  }
};

template <typename T>
T doit() {
 TEST_T<T> a;
 TEST_T<T> b;
 a.value = 1;
 b.value = 2;
 b.value = a.value;
 return b.value;
}

auto main() -> int
{
  std::cout << doit<int>() << '\n';
  return 0;
}
