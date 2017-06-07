#include <iterator>

template <typename iter_t>
void f(iter_t b, iter_t e) {
  return f(b, e, typename std::iterator_traits<iter_t>::iterator_category());
}

auto main() -> int
{
  return 0;
}
