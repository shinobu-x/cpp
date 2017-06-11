#include <iostream>
/*                       
 *                                     + <1, 0>
 *                                    /
 *                          + <2, 0> +
 *                         /          \
 *                        /            + <1, 1>
 *              + <4, 0> + 
 *             /          \            + <1, 2>
 *            /            \          /
 *           /              + <2, 2> +
 *          /                         \
 *         /                           + <1, 3>
 * <8, 0> +
 *         \                           + <1, 4>
 *          \                         /
 *           \              + <2, 4> +
 *            \            /          \
 *             \          /            + <1, 5>
 *              + <4, 4> +
 *                        \            + <1, 6>
 *                         \          /
 *                          + <2, 0> +
 *                                    \
 *                                     + <1, 7>
 */  
template <int N, int M>
struct index {};

template <typename T, int N, int M>
void zeroize_helper(T* const data, index<N, M>) {
  zeroize_helper(data, index<N/2, M>());
  zeroize_helper(data, index<N/2, M+N/2>());
}

template <typename T, int M>
void zeroize_helper(T* const data, index<1, M>) {
  data[M] = T();
}

template <typename T, int N>
void zeroize(T (&data)[N]) {
  zeroize_helper(data, index<N, 0>());
}

template <typename T, T N>
T doit() {
  T a[N];
  zeroize(a);
  for (T i = 0; i < N; ++i)
    std::cout << a[i] << '\n';
}
auto main() -> int
{
  doit<int, 8>();
  return 0;
}
