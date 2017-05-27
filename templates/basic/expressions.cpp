#include <iostream>

// Macro
#define sq_1(x) ((x)*(x))

// Function
double sq_2(double x) {
  return x*x;
}

// Function template
template <typename T>
T sq_3(const T& x) {
  return x*x;
}

// Class type1
class sq_4_t {
public:
  sq_4_t(double x) : s_(x*x) {}

  operator double() const {
    return s_;
  }
private:
  double s_;
};

// Class type2
class sq_5_t {
public:
  typedef double value_type;

  value_type operator()(value_type x) const {
    return x*x;
  }
};

// Template ver1
template <typename T>
class sq_6_t {
public:
  T operator()(T x) const {
    return x*x;
  }
};

// Template ver2
class sq_7_t {
public:
  template <typename T>
  T operator()(const T& x) const {
    return x*x;
  }
};

// Template ver3
template <typename T>
inline T sq_8(const T& x) {
  return x*x;
};

// Go!
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

  const sq_7_t sq_7;
  std::cout << sq_7(x) << '\n';

  std::cout << sq_8(x) << '\n';
}

auto main() -> int
{
  doit<double>();
  return 0;
}
