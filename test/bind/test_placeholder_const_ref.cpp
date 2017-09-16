#include <boost/bind/placeholders.hpp>
#include <boost/is_placeholder.hpp>

#include <cassert>

template <class T>
void do_test(T const&, int i) {
  assert(boost::is_placeholder<T>::value == i);
}

auto main() -> decltype(0) {
  do_test(boost::placeholders::_1, 1);
  do_test(boost::placeholders::_2, 2);
  do_test(boost::placeholders::_3, 3);
  do_test(boost::placeholders::_4, 4);
  do_test(boost::placeholders::_5, 5);
  do_test(boost::placeholders::_6, 6);
  do_test(boost::placeholders::_7, 7);
  do_test(boost::placeholders::_8, 8);
  do_test(boost::placeholders::_9, 9);
  return 0;
}
