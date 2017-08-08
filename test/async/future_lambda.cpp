#include <cassert>
#include <exception>
#include <future>
#include <iostream>
#include <string>

auto main() -> decltype(0) {
  const unsigned number_of_tests = 100;

  for (unsigned i = 0; i < number_of_tests; ++i)
    try {
      { // TEST1
        std::future<int> f1 = std::async(std::launch::async,
          [](){ return 123; });
        assert(f1.get() == 123);
      }

      { // TEST2
        std::future<int> f1 = std::async(std::launch::async,
          []() { return 123; });
        std::future<int> f2 = std::async(std::launch::async,
          [&f1](){ return 2*f1.get(); });
        assert(f2.get() == 246);
      }
    } catch (std::exception& e) {
      std::cout << e.what() << '\n';
    }

  return 0;
}
