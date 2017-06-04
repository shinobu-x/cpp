#include <iostream>

#define M_TOKEN(a, b, c) a##b##c

#define A1 1
#define A2 2
#define Z(a, b) a##b

#define X Y
#define Y X + Y

// X0(a) -> X1(a) -> X2(a, "a") -> const char* ca = "a"
#define X2(a, b) const char* c##a = b
#define X1(x) X2(x, #x)
#define X0(x) X1(x)

auto main() -> int
{
  std::cout << M_TOKEN(1, 2, 3) << '\n';
  std::cout << Z(A, 1) << '\n';
  std::cout << Z(A, 2) << '\n';
  X0(a);
  std::cout << ca << '\n';
}
