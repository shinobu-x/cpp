#include <boost/filesystem.hpp>

#include <iostream>

inline const char* macro_value(const char* name, const char* value) {
  static const char* no_value = "[no value]";
  static const char* not_defined = "[not defined]";

  BOOST_ASSERT_MSG(name, "name argument must not be a null pointer");
  BOOST_ASSERT_MSG(value, "value argument must not be a null pointer");

  return strcmp(name, value + 1)
    ? ((*value && *(value+1)) ? (value+1) : no_value) : not_defined;
}

#define MACRO_VALUE(x) macro_value(#x, BOOST_STRINGIZE(=x))

auto main() -> decltype(0) {
  std::cout << MACRO_VALUE(NOSUCMACRO) << '\n';
#define SUCHAMACRO
  std::cout << MACRO_VALUE(SUCHAMACRO) << '\n';
  std::cout << MACRO_VALUE(BOOST_VERSION) << '\n';
  std::cout << MACRO_VALUE(BOOST_FILESYSTEM_VERSION) << '\n';
  std::cout << MACRO_VALUE(BOOST_FIELSYSTEM_DEPRECATED) << '\n';
  std::cout << MACRO_VALUE(_MSC_VER) << '\n';
  std::cout << MACRO_VALUE(__MINGW32__) << '\n';
  return 0;
}
