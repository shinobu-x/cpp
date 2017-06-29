#include <algorithm>
#include <iostream>

template <typename T, T N>
struct static_parameter {};

template <typename T, T N>
struct static_value : static_parameter<T, N> {
  const static T value = N;
};

template <typename T>
struct zeroize_helper {
  static void apply(T* const data, static_value<int, 1>) {
    *data = T();
  }

  static void apply(T (&data)[2], static_value<int, 2>) {
    data[0] = data[1] = T();
  }

  static void apply(T* const data, const int N) {
    std::fill_n(data, N, T());
  }

  static T test(const int N) {
    return N;
  }
};

template <typename T, int N>
int zeroize(T (&data)[N]) {
  zeroize_helper<T>::apply(data, static_value<int, N>::value);
}

template <typename T>
T doit() {
  T a[5];
  zeroize<T, 5>(a);
}

auto main() -> int
{
  doit<int>();
}
