#include <iostream>

#define USE_V1 1

namespace X {
  namespace v1 {
    void f1() { std::cout << "v1 " << __func__ << '\n'; }
    void f2() { std::cout << "v1 " << __func__ << '\n'; }
  }

  namespace v2 {
    void f1() { std::cout << "v2 " << __func__ << '\n'; }
    void f2() { std::cout << "v2 " << __func__ << '\n'; }
  }

#ifdef USE_V1
  using namespace v1;
#else
  using namespace v2;
#endif
}

template <typename T>
T doit() {
  X::f1();
}

auto main() -> int
{
  doit<void>();
  return 0;
}
