#include <iostream>
#include <vector>

template <typename C, typename iter_t, typename base_t>
void erase_gap2(C& c, iter_t& i, iter_t (base_t::*)(iter_t)) {
  i = c.erase(i);
}

template <typename C, typename iter_t, typename base_t>
void erase_gap2(C& c, iter_t& i, void (base_t::*)(iter_t)) {
  c.erase(++i);
}

template <typename C>
void erase_gap(C& c, typename C::iterator& i) {
  erase_gap2(c, i, &C::erase);
}

template <typename C>
void do_print(C& c, typename C::iterator& i) {
  std::cout << "***************" << '\n';
  for (i=c.begin(); i!=c.end(); ++i)
    std::cout << *i << '\n';
  std::cout << "***************" << '\n';
}

template <typename T>
T doit() {
  std::vector<T> v;

  for (T i=0; i<10; ++i)
    v.push_back(i);

  typename std::vector<T>::iterator i;

  do_print(v, i);

  for (i=v.begin(); i!=v.end();) {
    erase_gap(v, i);
    ++i;
  }

  do_print(v, i);
}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
