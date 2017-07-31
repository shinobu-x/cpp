#include <future>

#include <iostream>

std::future<void> doit() {
  auto r = std::async([]{ std::cout << "1\n"; } );
  std::cout << "2\n";
  return r;
}

auto main() -> decltype(0) {
  doit().wait();
  std::cout << "3\n";
  return 0;
}
