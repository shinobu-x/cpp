/**
 *                min(j, a[k])
 * dp[i + 1][j] = Σ dp[i][j - k]
 *                k = 0
 *
 * dp[i + 1][j] =
 *   dp[i][j] + dp[i][j - 1] + dp[i][j - 2] + ... + dp[i][j - l(j)] # 1
 * // l = min(j, a[k])
 *
 * // dp[i + 1][j - 1]の状態
 * dp[i + 1][j - 1] =
 *   dp[i][j - 1] + dp[i][j - 2] + ... + dp[i][j - l(j - 1)] # 2
 *
