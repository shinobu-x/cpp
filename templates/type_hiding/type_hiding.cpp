#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>

template <typename IT, typename T>
void helper_f(IT begin, IT end, const T&) {

  typedef typename std::iterator_traits<IT>::value_type type_t;

}

template <typename IT>
void f(IT begin, IT end) {
  if (begin == end)
    return;

  helper_f(begin, end, *begin);
}

template <typename T>
void doit() {
  std::vector<T> v;
  for (T i=0; i<1000; ++i)
    v.push_back(i);

  f(v.begin(), v.end());
}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
