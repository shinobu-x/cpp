#include <map>
#include <iostream>
#include <functional>
#include <vector>

auto main() -> decltype(0) {
  std::vector<std::map<int, int> > v;

  std::map<int, int> m;

  for (int i; i < 1000; ++i) {
    m[i] = std::hash<int>()(i);
    v.push_back(m);
  }

  auto it = v.begin();
/*
  for (; it != v.end(); ++it) {
    auto i = it->begin();
    for (; i != it->end(); ++i)
      std::cout << i->first << " : " << i->second << '\n';
  }
*/
  for (auto& it : v)
    for (auto& [k, v] : it)
      std::cout << k << '\n';

  return 0;
}
