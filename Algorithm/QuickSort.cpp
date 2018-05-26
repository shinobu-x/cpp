#include <iostream>
#include <random>

template <typename T>
void swap(T* a, T* b) {
  T t = *a;
  *a = *b;
  *b = t;
}

template <typename T>
T Partition(T* a, T low, T high) {
  T p = a[high];
  T i = low - 1;

  for (T j = low; j <= high - 1; ++j) {
    if (a[j] <= p) {
      ++i;
      swap(&a[i], &a[j]);
    }
  }

  T index = i + 1;
  swap(&a[index], &a[high]);

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

  std::cout << "\n\n";

  QuickSort(arr, 0, N - 1);

  std::cout << "\n\n";

  for (int i = 0; i < N; ++i) {
    std::cout << arr[i] << ' ';
  }

  std::cout << "\n";

  return 0;
}
