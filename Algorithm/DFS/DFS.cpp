#include <iostream>

int n = 4;
int X[] = {1, 2, 4, 7};
int k = 18;

bool DFS1(int i, int s) {
  if (i == n) {
    return s == k;
  }

  if (DFS1(i + 1, s)) {
    return true;
  }

  if (DFS1(i + 1, s + X[i])) {
    return true;
  }

  return false;
}

auto main() -> decltype(0) {
  if (DFS1(0, 0)) {
    std::cout << "Found\n";
  }

  return 0;
}
