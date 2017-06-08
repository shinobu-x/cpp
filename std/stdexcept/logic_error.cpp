#include <cstdlib>
#include <stdexcept>
#include <iostream>

template <typename T>
T doit() {
  throw std::logic_error("Error");
}

auto main() -> int
{
  try {
    doit<int>();
  } catch (const std::logic_error& e) {
    // Exception
    std::cout << e.what() << '\n';
    std::exit(1);
  }
}
