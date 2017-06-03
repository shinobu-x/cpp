#include <iostream>
#include <string>
#include <map>
#include <random>

template <typename T>
T doit() {
  std::random_device rd;
  std::map<T, T> m;
  std::uniform_int_distribution<T> d(0, 9);

  for (T n = 0; n < 20000; ++n)
    ++m[d(rd)];

  for (auto v : m)
    std::cout << v.first << " : " << std::string(v.second/100, '*') << '\n';
}

auto main() -> int
{
  doit<int>();
  return 0;
}
