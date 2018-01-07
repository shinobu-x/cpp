#include <boost/variant.hpp>

#include <iostream>
#include <string>

auto main() -> decltype(0) {
  boost::variant<int, std::string, double> v;
  v = 1;
  if (v.type() == typeid(int)) {
    std::cout << "int" << '\n';
  }
  v = 1.0;
  if (v.type() == typeid(double)) {
    std::cout << "double" << '\n';
  }
}
