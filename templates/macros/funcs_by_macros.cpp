#include <iostream>

#define M_NS_BEGIN(x) namespace x {
#define M_NS_END      }

// ******

#define M_MAX(a, b) ((a) < (b) ? (a) : (b))
#define M_MIN(a, b) ((a) < (b) ? (b) : (a))
#define M_ABS(a)    ((a) < 0 ? -(a) : (a))
#define M_SQ(a)     ((a) * (a))

M_NS_BEGIN(TEST)
template <typename T, T N1, T N2>
struct TEST {
  static const T v1 = M_SQ(N1);
  static const T v2 = M_MAX(N1, N2);
  static const T v3 = M_MIN(N1, N2);
  static const T v4 = M_ABS(N1);
};
M_NS_END

template <typename T>
void doit() {
  TEST::TEST<T, 4, 5> a;

  std::cout << a.v1 << '\n';
}

auto main() -> int
{
  doit<int>();
  return 0;
}
