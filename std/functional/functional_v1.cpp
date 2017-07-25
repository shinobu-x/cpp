#include <functional>
#include <iostream>

template <typename T>
struct A {
  A(T a) : a_(a) {}
  T do_a(T i) { return i; };
private:
  T a_;
};

template <typename T>
void doit() {
  T m = 5;
  A<T> a(m);
  std::function<T(T)> f1 = [](T i) -> decltype(0) { return i; };
  std::function<T(T)> f2 = [&a](T i) -> decltype(0) {
    std::bind(&A<T>::do_a, &a, std::placeholders::_1, i, std::placeholders::_2); };
  std::cout << f1(1) << '\n';
  std::cout << f2(2) << '\n';

}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
