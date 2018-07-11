#include <iostream>
#include <queue>

#define MAX 1<<20
int N = 3;
int L[MAX] = {8, 5, 8};

auto main() -> decltype(0) {
  std::priority_queue<int, std::vector<int>, std::greater<int> > que;
  int ans = 0;
  for (int i = 0; i < N; ++i) {
    que.push(L[i]);
  }

  while (que.size() > 1) {
    auto l1 = que.top();
    que.pop();
    auto l2 = que.top();
    que.pop();
    ans += l1 + l2;
    que.push(l1 + l2);
  }

  std::cout << ans << "\n";

  return 0;
}
