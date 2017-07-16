#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <numeric>
#include <type_traits>

#include <immintrin.h>

/** -mavx for __m256 **/
struct timer {
public:
  timer() : start_(clock_::now()) {}

  void reset() {
    reset_();
  }

  friend std::ostream& operator<< (std::ostream& out, timer const &t) {
    return out << t.elapsed_().count() << "ms";
  }
private:
  typedef std::chrono::high_resolution_clock clock_;
  std::chrono::high_resolution_clock::time_point start_;

  void reset_() {
    start_ = clock_::now();
  }

  std::chrono::milliseconds elapsed_() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
       clock_::now() - start_);
  }
};

struct simd_cal_t {

  template <typename T, 
    typename std::enable_if<std::is_same<T, float>::value>::type* = nullptr>
  T do_simd_cal(size_t const n, T const* x) {
    const size_t p = 8;
    T sum[p];
    size_t const e = (n/p)*p;

    __m256 s = _mm256_setzero_ps();

    for (size_t i=0; i<e; i+=p)
      s = _mm256_add_ps(s, _mm256_loadu_ps(x+i));

    _mm256_storeu_ps(sum, s);

    for (size_t i=e; i<n; ++i)
      sum[0] += x[i];

    for (size_t i=1; i<sizeof(sum)/sizeof(T); ++i)
      sum[0] += sum[i];

    return sum[0];
  }

  template <typename T,
    typename std::enable_if<std::is_same<T, double>::value>::type* = nullptr>
  T do_simd_cal(size_t const n, T const* x) {
    const size_t p = 4;
    T sum[p];
    size_t const e = (n/p)*p;

    __m256d s = _mm256_setzero_pd();

    for (size_t i=0; i<e; i+=p)
      s = _mm256_add_pd(s, _mm256_loadu_pd(x+i));

    _mm256_storeu_pd(sum, s);

    for (size_t i=e; i<n; ++i)
      sum[0] += x[i];

    for (size_t i=1; i<sizeof(sum)/sizeof(T); ++i)
      sum[0] += sum[i];

    return sum[0];
  }
};

auto main() -> decltype(0)
{
  typedef float type_f;
  typedef double type_d;
  simd_cal_t simd_t;
  const size_t n = 10200000;
  srand(static_cast<unsigned>(time(NULL)));

  type_f* x = static_cast<type_f*>(malloc(sizeof(type_f)*n));
  type_d* y = static_cast<type_d*>(malloc(sizeof(type_d)*n));

  for (size_t i=0; i<n; ++i) {
    x[i] = static_cast<type_f>(rand())/RAND_MAX;
    y[i] = static_cast<type_d>(rand())/RAND_MAX;
  }
  timer t;
  t.reset();
  std::cout << std::accumulate(x, x+n, 0.0) << '\n';
  std::cout << t << '\n';
  t.reset();
  std::cout << simd_t.do_simd_cal<type_f>(n, x) << '\n';
  std::cout << t << '\n';
  t.reset();
  std::cout << std::accumulate(y, y+n, 0.0) << '\n';
  std::cout << t << '\n';
  t.reset();
  std::cout << simd_t.do_simd_cal<type_d>(n, y) << '\n';
  std::cout << t << '\n';
}
