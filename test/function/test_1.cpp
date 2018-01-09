#include <functional>
#include <iostream>

struct test_t {
  test_t(int i) : i(i) {}
  void test_f(const int j) const {  std::cout << i + j << '\n'; }
  int i;
};

void f(const int i) {
  std::cout << i << '\n';
}

auto main() -> decltype(0) {

  std::function<double(double, int)> f1 = [](double d, int i){ return d * i; };
  std::cout << f1(3.0, 2) << '\n';

  std::function<void(int)> f2 = f;
  f(5);

  std::function<void()> f3 = std::bind(f, 10);
  f3();

  std::function<void(const test_t&, int)> f4 = &test_t::test_f;
  test_t t(4);
  f4(t, 5);

  return 0;
}
