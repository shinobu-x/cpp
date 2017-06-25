#include <iostream>

template <typename T, T N>
T doit() {
  const T a=N;
  constexpr T b=*&a;
  [=]{
    constexpr T c=N;
    constexpr T b=*&c;
  };
}

auto main() -> int
{
  doit<size_t, 10>();
  return 0;
} 
