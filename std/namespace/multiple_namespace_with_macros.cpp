#include <iostream>

#define M_NS_BEGIN(X)  namespace X {
#define M_NS_END       }

M_NS_BEGIN(X);
M_NS_BEGIN(v1);
  void f1() { std::cout << "v1 " << __func__ << '\n'; }
M_NS_END;
M_NS_BEGIN(v2);
  void f1() { std::cout << "v1 " << __func__ << '\n'; }
M_NS_END;
M_NS_END;

template <typename T>
T doit() {
  X::v1::f1();
}

auto main() -> int
{
  doit<void>();
  return 0;
}
