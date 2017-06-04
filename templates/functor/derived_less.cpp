#include <functional>
#include <iostream>

template <typename T, typename less_t = std::less<T> >
class set : private less_t {
  inline bool less(const T& x, const T& y) const {
    return static_cast<const less_t&>(*this)(x, y);
  }
public:
  set(const less_t& l = less_t()) : less_t(l) {}

  void insert(const T& x, const T& y) {
    std::cout << (x < y) << '\n';
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
