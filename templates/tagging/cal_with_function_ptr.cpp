#include <iostream>

template <typename T, T VALUE>
struct static_parameter {};

template <typename T, T VALUE>
struct static_value : static_parameter<T, VALUE> {
  const static T value = VALUE;
};

enum binary_operation {
  sum, difference, product, division
};

#define M_SUM  x + y
#define M_DIFF x - y
#define M_PROD x * y
#define M_DIV  x / y

// ******

#define M_DEFINE(OPCODE, FORMULA)                                     \
                                                                      \
inline static_value<binary_operation, OPCODE> static_tag_##OPCODE() { \
                                                                      \
}                                                                     \
                                                                      \
template <typename T>                                                 \
T binary(T x, T y, static_value<binary_operation, OPCODE>) {          \
  return (FORMULA);                                                   \
}                                                                     \

M_DEFINE(sum, M_SUM);
M_DEFINE(difference, M_DIFF);
M_DEFINE(product, M_PROD);
M_DEFINE(division, M_DIV);

template <typename T, binary_operation OP>
inline T binary(T x, T y, static_value<binary_operation, OP> (*)()) {
  return binary(x, y, static_value<binary_operation, OP>());
}

template <typename T>
T doit() {
  T a1 = binary(8.0, 9.0, static_tag_product());
  std::cout << a1 << '\n';
}

auto main() -> int
{
  doit<double>();
  return 0;
}
