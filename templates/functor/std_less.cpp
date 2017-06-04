#include <functional>
#include <iostream>

template <typename T, typename less_t = std::less<T> >
class set {
  less_t less_;
public:
  set(const less_t& less = less_t()) : less_(less) {}

  void insert(const T& x, const T& y) {
    std::cout << (x < y) << '\n';
  }
};

template <typename T>
T doit() {
  set<T> a;
  T x = 1, y = 2;
  a.insert(x, y);
}

auto main() -> int
{
  doit<int>();
  return 0;
}
