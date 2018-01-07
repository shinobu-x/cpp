#include <boost/variant.hpp>

#include <iostream>
#include <string>

auto main() -> decltype(0) {
  boost::variant<int, std::string, double> v;

  std::cout << v.which() << '\n';
  v = 1;
  std::cout << v.which() << '\n';
  v = 1.0;
  std::cout << v.which() << '\n';
  return 0;
}
