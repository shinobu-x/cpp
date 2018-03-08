#include <boost/type_traits.hpp>

void doit() {
  static_assert(boost::is_same<
    boost::decay<int>::type, int>::value);
  static_assert(boost::is_same<
    boost::decay<char[2]>::type, char*>::value);
  static_assert(boost::is_same<
    boost::decay<char[2][3]>::type, char(*)[3]>::value);
  static_assert(boost::is_same<
    boost::decay<const char[2]>::type, const char*>::value);
  static_assert(boost::is_same<
    boost::decay<wchar_t[2]>::type, wchar_t*>::value);
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
