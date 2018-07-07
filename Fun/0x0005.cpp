#include <cmath>
#include <cstring>
#include <iostream>

#define INF 1000
int N = 4;
int w[INF] = {2, 1, 3, 2};
int v[INF] = {3, 2, 4, 2};
int W = 5;
// int dp[INF][INF];
int dp[INF];

int DoIt() {
  for (int i = 0; i < N; ++i) {
    for (int j = W; j >= w[i]; --j) {
      dp[j] = std::max(dp[j], dp[j - w[i]] + v[i]);
/*
      if (j < w[i]) {
        dp[i + 1][j] = dp[i][j];
      } else {
        dp[i + 1][j] = std::max(dp[i][j], dp[i + 1][j - w[i]] + v[i]);
      }
*/
/*
      for (int k = 0; k * w[i] <= j; ++k) {
        dp[i + 1][j] = std::max(dp[i + 1][j], dp[i][j - k * w[i]] + k * v[i]);
      }
*/
    }
  }
//  return dp[N][W];
  return dp[W];
}

auto main() -> decltype(0) {
//  memset(dp, -1, sizeof(dp));
  std::cout << DoIt() << "\n";
  return 0;
}
