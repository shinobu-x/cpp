#include "../HPP/CalElapsedTime.hpp"
#include <iostream>
#include <thread>

void doit() {
  std::this_thread::sleep_for(std::chrono::seconds(3));
}

auto main() -> decltype(0) {
  std::cout << CalElapsedTime<>::Execution(doit) << '\n';
  return 0;
}
