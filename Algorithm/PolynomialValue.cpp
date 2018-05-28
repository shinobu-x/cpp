#include <iostream>
#include <list>

template <typename I, typename R>
R PolynomialValue(I begin, I end, R x) {
  if (begin == end) {
    return R(0);
  }

  R sum(*begin);

  while (++begin != end) {
    sum *= x;
    sum += *begin;
  }

  return sum;
}

auto main() -> decltype(0) {
  std::list<int> l;

  for (int i = 0; i < 3; ++i) {
    l.push_back(i + 1);
  }

  auto begin = l.begin();
  auto end = l.end();

  // x^2 + 2 * x + 3
  std::cout << PolynomialValue(begin, end, 2) << "\n";

  return 0;
}
