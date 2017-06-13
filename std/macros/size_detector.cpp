#include <climits>
#include <iostream>

template <size_t S>
struct uint_n;

#define M_UINT_N(T, N) \
  template <> struct uint_n<N> { typedef T type; }

#define M_I32BIT 0xfffffffU
#define M_I16BIT 0xfffU
#define M_I8BIT  0xffU

#if (UCHAR_MAX == M_I8BIT)
M_UINT_N(unsigned char, 8);
#endif

#if (USHRT_MAX == M_I16BIT)
M_UINT_N(unsigned short, 16);
#elif (UINT_MAX == M_I16BIT)
M_UINT_N(unsigned int, 16);
#endif

#if (UINT_MAX == M_I32BIT)
M_UINT_N(unsigned int, 32)
#elif (ULONG_MAX == M_I32BIT)
M_UINT_N(unsigned long, 32);
#endif

template <typename T>
T doit() {

}
auto main() -> int
{
  std::cout << UINT_MAX << '\n';
  return 0;
}
