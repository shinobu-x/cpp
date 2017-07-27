#include <algorithm>
#include <iostream>
#include <vector>

template <typename T>
struct type_t {
  type_t(std::vector<T> a) : a_(a) {}
  type_t(type_t const &that) { a_ = that.a_; }
  ~type_t(){}

  void push(T n) {
    a_.push_back(n);
  }

private:
  std::vector<T> a_;
};

template <typename T>
void doit() {
  std::vector<T> v;
  type_t<T> a(v);
  type_t<T> b(a);
}

auto main() -> decltype(0) {
  doit<int>();
  return 0;
}
