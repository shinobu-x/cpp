#include <iostream>
#include <random>
#include <utility>
/**
 * クイックソート
 *
 * 分割統治法の一種
 * 最悪計算量：O(n^2)
 * 最良計算量：O(n log n)
 * 平均計算時間：O(n log n)
 * 最悪空間計算量：O(n)
 * 
 * アルゴリズム
 * 1) 適当な数（ピボット）を選択
 * 2) ピボットより小さい数を左、大きい数を右に移動
 * 3) 二分割された各々のデータを、それぞれソート
 *                                                               参考：Wikipedia
 */
template <typename T>
T Partition(T* a, T low, T high) {
  T p = a[high];
  T i = low - 1;

  for (T j = low; j <= high - 1; ++j) {
    if (a[j] <= p) {
      ++i;
      std::swap(a[i], a[j]);
    }
  }

  T index = i + 1;
  std::swap(a[index], a[high]);

  return index;
}

template <typename T>
void QuickSort(T* a, T low, T high) {
  if (low < high) {
    auto p = Partition(a, low, high);
    QuickSort(a, low, p - 1);
    QuickSort(a, p + 1, high);
  }
}

auto main() -> decltype(0) {
  int N = 20;
  int arr[N];

  std::random_device rnd;
  std::mt19937 mt(rnd());
  std::uniform_int_distribution<> rndint(0, 99);

  for (int i = 0; i < N; ++i) {
    arr[i]  = rndint(mt);
  }

  for (int i = 0; i < N; ++i) {
    std::cout << arr[i] << ' ';
  }
  std::cout << "\n";
  QuickSort(arr, 0, N - 1);
  for (int i = 0; i < N; ++i) {
    std::cout << arr[i] << ' ';
  }

  std::cout << "\n";

  return 0;
}
