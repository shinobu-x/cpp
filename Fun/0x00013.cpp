#include <iostream>

#define INF 1000
int n = 3;
int m = 3;
int a[INF] = {1, 2, 3};
int M = 10000;
int dp[INF][INF];

auto main() -> decltype(0) {
  for (int i = 0; i <= n; ++i) {
    dp[i][0] = 1;
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 1; j <= m; ++j) {
      if (j - 1 - a[i] >= 0) {
        dp[i + 1][j] =
          (dp[i + 1][j - 1] + dp[i][j] - dp[i][j - 1 - a[i]] + M) % M;
      } else {
        dp[i + 1][j] = (dp[i + 1][j - 1] + dp[i][j]) % M;
      }
    }
  }
  std::cout << dp[n][m] << "\n";
  return 0;
}
