#include <functional>
#include <iostream>

template <typename T>
struct type_t {
  type_t(T a) : a_(a) {}

  T get() {
    return a_;
  }

  T set(T a) {
    a_ = a;
    return a_;
  }

private:
  T a_;
};

template <typename T>
T f(T a, T b) {
  return a+b;
}

template <typename T, T N>
T doit() {
  type_t<T> t1(N);
  auto t2 = type_t<T>(4);
  auto f1 = std::bind(f<T>, std::placeholders::_1, std::placeholders::_2);
  auto f2 = std::bind(&type_t<T>::get, &t1);
  auto f3 = std::bind(&type_t<T>::set, &t1, std::placeholders::_1);
  std::cout << f1(1, 2) << '\n';
  std::cout << f2() << '\n';
  std::cout << f3(2) << '\n';
}

auto main() -> int
{
  doit<int, 1>();
  return 0;
}  
