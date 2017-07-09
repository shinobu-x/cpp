#include <algorithm>
#include <iostream>

struct yes_type {
  enum { value = true };
};

struct no_type {
  enum { value = false };
};

struct swap_traits {
  template <typename T, void (T::*F)(T&)>
  class yes1 : public yes_type {};

  template <typename T, void (*F)(T&, T&)>
  class yes2 : public yes_type {};

  template <typename T>
  inline static void apply(T& a, T& b) {
    apply1(a, b, test(&a));
  }

private:
  template <typename T>
  static yes1<T, &T::swap>* test(T*) {
    return 0;
  }

  template <typename T>
  static yes2<T, &T::swap>* test(T*) {
    return 0;
  }

  static no_type* test(void*) {
    return 0;
  }

private:
  template <typename T>
  inline static void apply1(T& a, T& b, no_type*) {
    std::swap(a, b);
  }

  template <typename T>
  inline static void apply2(T& a, T& b, yes_type*) {
    apply2(a, b, &T::swap);
  }

  template <typename T>
  inline static void apply2(T& a, T& b, void (*)(T&, T&)) {
    T::swap(a, b);
  }

  template <typename T, typename BASE>
  inline static void apply2(T& a, T& b, void (BASE::*)(BASE&)) {
    a.swap(b);
  }

  template <typename T>
  inline static void apply2(T& a, T& b, ...) {
    std::swap(a, b);
  }
};

template <typename T>
void doit() {
  T a=1, b=2;
  swap_traits s;
  std::cout << a << " " << b << '\n';
  s.apply<T>(a, b);
  std::cout << a << " " << b << '\n';
}
auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
