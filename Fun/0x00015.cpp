#include <iostream>
#include <queue>

#define MAX 1000

// ガソリンスタンド
int N = 4;                     // 数
int A[MAX] = {10, 14, 20, 21}; // 位置
int B[MAX] = {10, 5, 2, 4};    // 補給量

int P = 10; // 燃料初期値
int L = 25; // 移動距離

auto main() -> decltype(0) {
  std::priority_queue<int> que;
  int ans = 0;
  int pos = 0;
  int fuel = P;

  for (int i = 0; i < N; ++i) {
    int d = A[i] - pos; // 補給地点までの移動距離

    while (fuel - d < 0) {
      if (que.empty()) {
        std::cout << "-1\n";
        return 0;
      }

      fuel += que.top();
      que.pop();
      ++ans;
    }

    fuel -= d; // 移動距離に対する燃料消費量
    pos = A[i]; // 移動後の位置
    que.push(B[i]); // 次の補給地点を登録
  }
  std::cout << ans << "\n";
  return 0;
}
