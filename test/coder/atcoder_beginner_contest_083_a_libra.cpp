#include <cstdlib>
#include <iostream>

auto main() -> decltype(0) {
  int w[4];
  for (int i = 0; i < 3; ++i)
    w[i] = std::rand() % 10 + 1;

  if ((w[0] + w[1]) > (w[2] + w[4])) {
    std::cout << "Left\n";
  } else if ((w[0] + w[1]) < (w[2] + w[4])) {
    std::cout << "Right\n";
  } else {
    std::cout << "Balanced\n";
  }

  return 0;
}
