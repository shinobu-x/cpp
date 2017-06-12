#include <algorithm>
#include <iostream>
#include <vector>

template <typename T>
const T& min(const std::vector<T>& v) {
  typedef typename std::vector<T>::const_iterator it;
  it select = v.begin();
  it next = std::next(select);
  it end = v.end();

  for (; next != end; ++next)
    if (*next < *select) select = next;

  return *select;
};

template <typename T>
T doit() {
  T a[5] = {1, 2, 3, 4, 5};
  std::vector<T> v;

  for (T i = 0; i < sizeof(a)/sizeof(T); ++i)
    v.push_back(a[i]);

  T r = min(v);

  std::cout << r << '\n';
}

auto main() -> int
{
  doit<int>();
  return 0;
}
