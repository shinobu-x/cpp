#include <iostream>

template <size_t N>
struct fixed_size {
  typedef char type[N];
};

template <typename T>
T doit() {
  fixed_size<3>::type& f();
  T a = sizeof(f());
  std::cout << a << '\n';
}

auto main() -> int
{
  doit<int>();
  return 0;
}
