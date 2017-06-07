#include <iterator>
#include <iostream>
#include <list>
#include <vector>

template <typename T>
T doit() {
  std::vector<T> v = {1, 2, 3, 4};
  std::cout << std::distance(v.begin(), v.end()) << '\n';
  std::list<T> ls = {1, 2, 3, 4, 5, 6};
  std::cout << std::distance(ls.begin(), ls.end()) << '\n';
}

auto main() -> int
{
  doit<int>();
  return 0;
}
