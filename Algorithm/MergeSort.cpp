#include <iostream>
/**
 * 計算量: O(n log n)
 *
 * アルゴリズムの動作例
 *
 * 1) データの分割フェーズ
 *   8 4 3 7 6 5 2 1
 *
 *   8 4 3 7 | 6 5 2 1
 *
 *   8 4 | 3 7 | 6 5 | 2 1
 *
 * 2) 2以下になった時点でソート＆マージ
 *   4 8 | 3 7 | 5 6 | 1 2
 *
 *   4 8 3 7 | 5 6 1 2 // マージ
 *
 *   3 4 7 8 | 1 2 5 6 // ソート
 *
 *   1 2 3 4 5 6 7 8
 *                   Wikipediaより
 */
int n = 8;
int a[8] = {12, 9, 15, 3, 8, 17, 6, 1};

void Merge(int* a, int left, int mid, int right) {
  // 左側を一時領域に保存
  int nl = mid - left;
  int l[nl];
  for (int i = 0; i < nl; ++i) {
    l[i] = a[left + i]; // e.g., 0 + 0, 0 + 1, ..., 0 + nl - 1
  }

  // 右側を一時領域に保存
  int nr = right - mid;
  int r[nr];
  for (int i = 0; i < nr; ++i) {
    r[i] = a[mid + i]; // e.g., 4 + 0, 4 + 1, ..., 4 + nr - 1
  }

  int lc = 0;
  int rc = 0;
  int i = left;
  while (lc < nl && rc < nr) {
    // 左側が小さい場合
    if (l[lc] <= r[rc]) {
      a[i] = l[lc];
      ++lc;
    } else { // 右側が小さい場合
      a[i] = r[rc];
      ++rc;
    }
    ++i;
  }

  // 一時配列の残りのデータを移動
  // 左側
  while (lc < nl) {
    a[i] = l[lc];
    ++i;
    ++lc;
  }

  // 右側
  while (rc < nr) {
    a[i] = r[rc];
    ++i;
    ++rc;
  }
}

void MergeSort(int* a, int left, int right) {
  if (right - left == 1) {
    return;
  }
  // +leftはオーバーフロー対策
  int mid = left + (right - left) / 2;
  MergeSort(a, left, mid);
  MergeSort(a, mid, right);
  Merge(a, left, mid, right);
}

auto main() -> decltype(0) {
  /*
  int n;
  std::cin >> n;
  int a[n];
  for (int i = 0; i < n; ++i) {
    std::cin >> a[i];
  }
  */
  MergeSort(a, 0, n);
  for (int i = 0; i < n; ++i) {
    std::cout << a[i] << "\n";
  }
  return 0;
}
