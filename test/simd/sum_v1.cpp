#include <cstdlib>
#include <ctime>
#include <iostream>
#include <numeric>

#include <immintrin.h>

/** -mavx for __m256 **/

template <typename T>
struct base_t {
  typedef T type_t;

  size_t get_num() const {
    if (sizeof(T) == 4)
      return 8;
    else
      return 4;
  }
};

template <typename T>
struct simd_sum; // Nothing

template <>
struct simd_sum<float> : base_t<float> {
  const type_t operator() (size_t const n, type_t const* x) {
    const size_t p = get_num();
    type_t sum[p];
    size_t const e = (n/p)*p;

    __m256 s = _mm256_setzero_ps();

    for (size_t i=0; i<e; i+=p)
      s = _mm256_add_ps(s, _mm256_loadu_ps(x+i));

    _mm256_storeu_ps(sum, s);

    for (size_t i=e; i<n; ++i)
      sum[0] += x[i];

    for (size_t i=1; i<sizeof(sum)/sizeof(type_t); ++i)
      sum[0] += sum[i];

    return sum[0];
  }
};

template <>
struct simd_sum<double> : base_t<double> {
  const type_t operator() (size_t const n, type_t const* x) {
    const size_t p = get_num();
    type_t sum[p];
    size_t const e = (n/p)*p;

    __m256d s = _mm256_setzero_pd();

    for (size_t i=0; i<e; i+=p)
      s = _mm256_add_pd(s, _mm256_loadu_pd(x+i));

    _mm256_storeu_pd(sum, s);

    for (size_t i=e; i<n; ++i)
      sum[0] += x[i];

    for (size_t i=1; i<sizeof(sum)/sizeof(type_t); ++i)
      sum[0] += sum[i];

    return sum[0];
  }
};

auto main() -> decltype(0)
{
  typedef float type_f;
  typedef double type_d;
  simd_sum<type_f> simd_f;
  simd_sum<type_d> simd_d;

  const size_t n = 102;
  srand(static_cast<unsigned>(time(NULL)));

  type_f* x = static_cast<type_f*>(malloc(sizeof(type_f)*n));
  type_d* y = static_cast<type_d*>(malloc(sizeof(type_d)*n));

  for (size_t i=0; i<n; ++i) {
    x[i] = static_cast<type_f>(rand())/RAND_MAX;
    y[i] = static_cast<type_d>(rand())/RAND_MAX;
  }

  std::cout << std::accumulate(x, x+n, 0.0) << '\n';
  std::cout << simd_f(n, x) << '\n';
  std::cout << std::accumulate(y, y+n, 0.0) << '\n';
  std::cout << simd_d(n, y) << '\n';
}
