#include <iostream>
#include <stdexcept>

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
T binary(T x, T y, static_value<binary_operation, OP> (*)()) {
  switch (OP) {
  case sum:
    return M_SUM;
  case difference:
    return M_DIFF;
  case product:
    return M_PROD;
  case division:
    return M_DIV;
  default:
    throw std::runtime_error("Invalid operation");
  }
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
