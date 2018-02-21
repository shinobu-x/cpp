#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

auto main() -> decltype(0) {
  std::vector<double> v{3.0, 1.0, 0.2};
  auto max = std::max_element(v.begin(), v.end());
  std::cout << *max << '\n';

  return 0;
}
