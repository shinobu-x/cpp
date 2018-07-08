#include  <cstring>
#include <iostream>

#define INF 1000
int n = 4;
int m = 3;
int M = 10000;
int dp[INF][INF];

auto main() -> decltype(0) {
//  memset(dp, -1, sizeof(dp));
  dp[0][0] = 1;
  for (int i = 1; i <= m; ++i) {
    for (int j = 0; j <= n; ++j) {
      if (j - i >= 0) {
        dp[i][j] = (dp[i - 1][j] + dp[i][j - i]) % M;
      } else {
        dp[i][j] = dp[i - 1][j];
      }
    }
  }
  std::cout << dp[m][n] << "\n";
  return 0;
}
