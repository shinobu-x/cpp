#include <boost/version.hpp>
#include <iostream>

auto main() -> decltype(0) {
  std::cout << BOOST_LIB_VERSION << '\n';
  return 0;
}
