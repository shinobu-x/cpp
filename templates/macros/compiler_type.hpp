#include <iostream>

struct msvc {};
struct gcc {};

#if defined(__MSVC)
typedef msvc compiler_type
#elif defined(__GCC__)
typedef gcc compiler_type
#endif

template <typename scalar_t, typename compiler_type>
struct maths {
  static scalar_t multiplied_by_two(const scalar_t x) {
    return 2*x;
  }
};

template <>
struct maths<unsigned int, msvc> {
  static unsigned int multiplied_by_two(const unsigned int x) {
    return x << 1;
  }
};

template <typename scalar_t>
inline scalar_t multiplied_by_two(const scalar_t& x) {
  return maths<scalar_t, compiler_type>::multiplied_by_two(x);
}
