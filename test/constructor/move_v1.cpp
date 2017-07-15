#include <iostream>
#include <utility>
#include <vector>

template <typename T>
struct type_t {
  type_t(T a);
  type_t(type_t const& that);
  type_t(type_t&& that);
  ~type_t();

  friend std::ostream& operator<< (std::ostream& out, type_t const &t) {
    return out << t.a_;
  }

private:
  T a_;
};

template <typename T>
type_t<T>::type_t(T a) : a_(a) {
  std::cout << "Constructor" << '\n';
}

template <typename T>
type_t<T>::type_t(type_t const& that) {
  std::cout << "Copy constructor" << '\n';
}

template <typename T>
type_t<T>::type_t(type_t&& that) {
  std::cout << "Move constructor" << '\n';
  a_ = that.a_;
}

template <typename T>
type_t<T>::~type_t() {
  std::cout << "Destructor" << '\n';
}

template <typename T>
void doit() {
  T x = 2;
  type_t<T> a(x);
  type_t<T> b(a);
  type_t<T> c = std::move(b);
  std::cout <<  c << '\n';
}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
