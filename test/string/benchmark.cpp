#include <chrono>
#include <iostream>
#include <string>
#include <string_view>

const static int n = 10000000;

template <typename F>
void do_bench(F&& f) {
  std::cout << "====== start ======\n";
  auto s = std::chrono::steady_clock::now();
  f();
  auto e = std::chrono::steady_clock::now();
  std::chrono::duration<double> d = e - s;
  std::cout << d.count() << '\n';
  std::cout << "====== end ======\n";
}

void do_string() {
  std::string s = "abcdefghijklmnopqrstuvwxyz";
//  std::string a;
  for (int i = 0; i < n; ++i)
    auto a = s.substr(4);
//  std::cout << a << '\n';
}

void do_string_view() {
  std::string_view sv = "abcdefghijklmnopqrstuvwxyz";
//  std::string a;
  for (int i = 0; i < n; ++i)
    auto a = sv.substr(4);
//  std::cout << a << '\n';
}

void s_to_s() {
  std::string s1 = "abcdefghijklmnopqrstuvwxyz";
//  std::string s2;
  for (int i = 0; i < n; ++i)
    auto s2 = s1.substr(0, 6);
//  std::cout << s2 << '\n';
}

void sv_to_s() {
  std::string s1 = "abcdefghijklmnopqrstuvwxyz";
//  std::string s2;
  for (int i = 0; i < n; ++i)
    auto s2 = std::string_view{s1}.substr(0, 6);
//  std::cout << s2 << '\n';
}

auto main() -> decltype(0) {
  do_bench(do_string);
  do_bench(do_string_view);
  do_bench(s_to_s);
  do_bench(sv_to_s);
  return 0;
}
