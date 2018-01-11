#include <chrono>
#include <iostream>
#include <string>
#include <string_view>

const static int n = 10000000;

void do_string() {
  std::string s = "abcdefghijklmnopqrstuvwxyz";
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < n; ++i)
    auto a = s.substr(4);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> d = end - start;
  std::cout << s.substr(4) << '\n';
  std::cout << d.count() << '\n';
}

void do_string_view() {
  std::string_view sv = "abcdefghijklmnopqrstuvwxyz";
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < n; ++i)
    auto a = sv.substr(4);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> d = end - start;
  std::cout << sv.substr(4) << '\n';
  std::cout << d.count() << '\n';
}

auto main() -> decltype(0) {
  do_string();
  do_string_view();
  return 0;
}
