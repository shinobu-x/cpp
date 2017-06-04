#include <functional>
#include <iostream>

template <typename T>
struct less {
  static bool apply(const T& x, const T& y) {
    return x < y;
  }
};

template <typename T, typename less_t = less<T> >
class set {
public:
  void insert(const T& x, const T& y) {
    std::cout << less_t::apply(x, y) << '\n';
  }
};

template <typename T>
T doit() {
  set<T> l;
  T a = 1, b = 2;
  l.insert(a, b);
}

auto main() -> int
{
  doit<int>();
  return 0;
}
