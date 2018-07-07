#include <algorithm>
#include <iostream>

#define INF 1000
int N = 4;
int W = 5;
int w[INF] = {3, 4, 2};
int v[INF] = {4, 5, 3};
int dp[INF][INF];

int DoIt() {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N * 5; ++j) {
      if (j < v[i]) {
        dp[i + 1][j] = dp[i][j];
      } else {
        dp[i + 1][j] = std::min(dp[i][j], dp[i][j - v[i]] + w[i]);
      }
    }
  }

  int Res = 0;
  for (int i = 0; i < N * 5; ++i) {
    if (dp[N][i] <= W) {
      Res = i;
    }
  }

  return Res;
}

auto main() -> decltype(0) {
  std::cout << DoIt() << "\n";
  return 0;
}
