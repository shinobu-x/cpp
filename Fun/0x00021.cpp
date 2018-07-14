#include <cmath>
#include <iostream>

#define MAX 2000
int N = 4;
int H = 5;
int A = 100;
int B = 4;
int C = 60;
int D = 1;
int E = 4;
int dp[MAX][MAX];

auto main() -> decltype(0) {
  int j_max = H + B * N;
  for (int i = 0; i < N + 1; ++i) {
    for (int j = 0; j < j_max; ++j) {
      dp[i][j] = MAX;
    }
  }
  dp[0][H] = 0;
  for (int i = 0; i < N; ++i) {
    for (int j = 1; j < j_max; ++j) {
      if (i + 1 <= N) {
        if (j + B < j_max) {
          dp[i + 1][j + B] = std::min(dp[i + 1][j + B], dp[i][j] + A);
        }
        if (j + D < j_max) {
          dp[i + 1][j + D] = std::min(dp[i + 1][j + D], dp[i][j] + C);
        }
        if (j - E > 0) {
          dp[i + 1][j - E] = std::min(dp[i + 1][j - E], dp[i][j] + 0);
        }
      }
    }
  }
  int sum = MAX;
  for (int j = 1; j < j_max; ++j) {
    sum = std::min(sum, dp[N][j]);
  }
  std::cout << sum << "\n";
  return 0;
}
