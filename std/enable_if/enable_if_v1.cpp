#include <iostream>
#include <type_traits>

extern void* enabler;

// template <typename T>
struct selector_t {
/*
  template <typename T, typename std::enable_if<
    std::is_same<T, float>::value>::type*& = enabler>
  void type_f() {
    std::cout << "Type float" << '\n';
  }

  template <typename T, typename std::enable_if<
    std::is_same<T, double>::value>::type* = nullptr>
  void type_f() {
    std::cout << "Type double" << '\n';
  }
*/
  template <typename T>
  typename std::enable_if<std::is_same<T, double>::value>::type type_f() {
    std::cout << "Type double" << '\n';
  }

  template <typename T>
  typename std::enable_if<std::is_same<T, float>::value>::type type_f() {
    std::cout << "Type float" << '\n';
  }

  template <typename T>
  typename std::enable_if<std::is_same<T, int>::value>::type type_f() {
    std::cout << "Type int" << '\n';
  }
};

auto main() -> decltype(0)
{
  selector_t t;
  t.type_f<float>();
  t.type_f<double>();
  t.type_f<int>();
  return 0;
}

