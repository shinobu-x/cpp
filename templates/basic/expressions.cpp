#include <iostream>

#define sq_1(x) ((x)*(x))

double sq_2(double x) {
  return x*x;
}

template <typename T>
T sq_3(const T& x) {
  return x*x;
}

class sq_4_t {
public:
  sq_4_t(double x) : s_(x*x) {}

  operator double() const {
    return s_;
  }
private:
  double s_;
};

class sq_5_t {
public:
  typedef double value_type;

  value_type operator()(value_type x) const {
    return x*x;
  }
};

template <typename T>
class sq_6_t {
public:
  T operator()(T x) const {
    return x*x;
  }
};

template <typename T>
inline T sq_7(const T& x) {
  return x*x;
};

template <typename T>
void doit() {
  T x = 3.14;

  std::cout << sq_1(x) << '\n';
  std::cout << sq_2(x) << '\n';
  std::cout << sq_3(x) << '\n';

  // ******

  const sq_4_t sq_4(x);
  std::cout << sq_4 << '\n';

  const sq_5_t sq_5;
  std::cout << sq_5(x) << '\n';

  // ******

  const sq_6_t<T> sq_6;
  std::cout << sq_6(x) << '\n';

  std::cout << sq_7(x) << '\n';
}

auto main() -> int
{
  doit<double>();
  return 0;
}
