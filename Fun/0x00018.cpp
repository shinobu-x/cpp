#include <iostream>
#include <vector>

auto main() -> decltype(0) {
  int n = 5;
  auto x = 1<<n;
  // std::cout << x << "\n";

  for (int i = 0; i < x; ++i) {
    for (int j = 0; j < n; ++j) {
      auto y = 1<<j;
      // std::cout << y << "\n";
      if (i & y) {
        std::cout << y << "\n";
      }
    }
  }
  return 0;
}
