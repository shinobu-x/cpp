#include <boost/filesystem/operations.hpp>

#include <iostream>
#include <exception>

auto main(int argc, char** argv) -> decltype(0) {
//  boost::filesystem::path::default_name_check(boost::filesystem::native);

  if (argc != 3) {
    std::cout << "Usage: <program name> <path1> <path2>\n";
    return 2;
  }

  bool is_equivalent;

  try {
    is_equivalent = boost::filesystem::equivalent(*(argv+1), *(argv+2));
  } catch(const std::exception& e) {
    std::cout << e.what() << '\n';
    return 3;
  }

  std::cout << (is_equivalent ? "Equivalent\n" : "Not equivalent\n");

  return 0;

  return 0;
}
