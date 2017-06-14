#include <iostream>

#define M_SQ(a) ((a)*(a))

template <size_t M, size_t N>
struct static_raise;

template <size_t M>
struct static_raise<M, 0> {
  static const size_t value = 1;
};

template <size_t M, size_t N>
struct static_raise {
private:
  static const size_t v = static_raise<M, N/2>::value;
public:
  static const size_t value = M_SQ(v)*(N % 2 ? M : 1);
};

template <typename T>
T doit() {
  const T a = 5, b = 4;
  std::cout << static_raise<a, b>::value << '\n';
}

auto main() -> int
{
  doit<size_t>();
  return 0;
}
