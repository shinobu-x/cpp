#include <iostream>

/**
 * Initialize given array with zero.
 */

template <typename T, T VALUE>
struct static_parameter {};

template <typename T, T VALUE>
struct static_value : static_parameter<T, VALUE> {
  const static T value = VALUE;
};

template <typename T, int N>
void zeroize_helper(T* const data, static_value<int, N>) {
  zeroize_helper(data, static_value<int, N-1>());
  data[N-1] = T();
}

/**
template <typename T, int N>
void zeroize_helper(T* const data, static_value<int, N> {
  data[N-1] = T();
  zeroize_helper(data, static_value<int, N-1>());
}
*/

template <typename T>
void zeroize_helper(T* const data, static_value<int, 1>) {
  data[0] = T();
}

template <typename T, int N>
void zeroize(T (&data)[N]) {
  zeroize_helper(data, static_value<int, N>());
}

template <typename T, T N>
T doit() {
  T a[N];
  zeroize<T, N>(a);

  for (int i = 0; i < N; ++i)
    std::cout << a[i] << '\n';
}
auto main() -> int
{
  doit<int, 10>();
  return 0;
}
