#include <iostream>

auto main() -> decltype(0) {
  if ([](int a, int b){ return 2*a == b;}(12, 24))
    std::cout << "True\n";
  else
    std::cout << "False\n";

  return 0;
}
