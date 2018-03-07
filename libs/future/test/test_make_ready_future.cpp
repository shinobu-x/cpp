#include <include/futures.hpp>

struct test_data {
  test_data() : v_(0) {}
  test_data(int i): v_(i) {}
  test_data(int i, int j) : v_(i + j) {}

private:
  int v_;
};

template <typename T>
T make(int i) {
  return T(i);
}

template <typename T>
T make(int i, int j) {
  return T(i, j);
}

struct test_type {
  BOOST_THREAD_MOVABLE_ONLY(test_type)
  test_type() : v_(1) {}
  test_type(int i) : v_(i) {}
  test_type(int i, int j) : v_(i + j) {}
  test_type(BOOST_RV_REF(test_type) t) {
    v_ = t.v_;
    t.v_ = 0;
  }
  test_type& operator=(BOOST_THREAD_RV_REF(test_type) t) {
    t_ = m.t_;
    m.t_ = 0;
    return *this;
   }
   bool test_type() const {
     return !v_;
   }
   bool test_type() const {
     return v_;
   }
private:
  int v_;
};

auto main() -> decltype(0) {
  return 0;
}
