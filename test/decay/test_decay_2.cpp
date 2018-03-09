#include <boost/type_traits.hpp>

void doit() {
  static_assert(boost::is_same<
    boost::decay<int>::type, int>::value);
  static_assert(!boost::is_same<
    boost::decay<int&>::type, int&>::value);
  static_assert(!boost::is_same<
    boost::decay<const int>::type, const int>::value);
  static_assert(!boost::is_same<
    boost::decay<int const&>::type, int const&>::value);
  static_assert(!boost::is_same<
    boost::decay<volatile int>::type, volatile int>::value);
  static_assert(!boost::is_same<
    boost::decay<int volatile&>::type, int volatile&>::value);
  static_assert(boost::is_same<
    boost::decay<char[2]>::type, char*>::value);
  static_assert(boost::is_same<
    boost::decay<char[2][3]>::type, char(*)[3]>::value);
  static_assert(boost::is_same<
    boost::decay<const char[2]>::type, const char*>::value);
  static_assert(boost::is_same<
    boost::decay<wchar_t[2]>::type, wchar_t*>::value);
  static_assert(boost::is_same<
    boost::decay<wchar_t[2][2]>::type, wchar_t(*)[2]>::value);
  static_assert(boost::is_same<
    boost::decay<const wchar_t[2]>::type, const wchar_t*>::value);
  typedef int value_type1(void);
  typedef int value_type2(int);
  static_assert(boost::is_same<
    boost::decay<value_type1>::type, int (*)(void)>::value);
  static_assert(boost::is_same<
    boost::decay<value_type2>::type, int (*)(int)>::value);
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
