#include <iostream>
#include <iterator>
#include <memory>
#include <vector>

template <typename T>
void doit() {
  std::vector<std::unique_ptr<T> > v;
  for (T i=0; i<10; ++i)
    v.emplace_back(new T(i));

  for (typename decltype(v)::iterator it=v.begin(); it!=v.end(); ++it) {
    auto it_ = std::make_move_iterator(it);
    std::unique_ptr<T> p = *it_;
    std::cout << *p << '\n';
  }
}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
