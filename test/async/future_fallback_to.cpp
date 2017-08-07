#include <cassert>
#include <exception>
#include <future>
#include <iostream>
#include <string>
#include <utility>

#include "../macro/config.hpp"

const unsigned number_of_tests = 200;

int func_ex() {
  throw std::logic_error("Error");
}

int func() {
  return 1;
}

void test_1() {
  for (unsigned i = 0; i < number_of_tests; ++i)
    try {
      std::future<int> f1 = std::async(std::launch::async, &func);
      f1.wait();
      assert(f1.get() == 1);
    } catch (std::exception& e) {
      LOG;
      std::cout << e.what() << '\n';
      return;
    } catch (...) {
      LOG;
      return;
    }
}

void test_2() {
  for (unsigned i = 0; i < number_of_tests; ++i)
    try {
      std::future<int> f1 = std::async(&func);
      std::future<int> f2 = std::move(f1);
      f2.wait();
      assert(f2.get() == 1);
    } catch (std::exception& e) {
      LOG;
      std::cout << e.what() << '\n';
      return;
    } catch (...) {
      LOG;
      return;
    }
}

auto main() -> decltype(0) {
  test_1(); test_2();
  return 0;
}
