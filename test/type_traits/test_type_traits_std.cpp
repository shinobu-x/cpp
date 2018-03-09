#include <type_traits>

void doit() {
  static_assert(std::is_same<
    std::add_const_t<int>, int const>::value);
  static_assert(std::is_same<
    std::add_cv_t<int>, int const volatile>::value);
  static_assert(std::is_same<
    std::add_lvalue_reference_t<int>, int&>::value);
  static_assert(std::is_same<
    std::add_pointer_t<int>, int*>::value);
  static_assert(std::is_same<
    std::add_rvalue_reference_t<int>, int&&>::value);
  static_assert(std::is_same<
    std::add_volatile_t<int>, int volatile>::value);
  static_assert(std::is_same<
    std::common_type_t<char, short>, int>::value);
  static_assert(std::is_same<
    std::conditional_t<true, char, short>, char>::value);
  static_assert(std::is_same<
    std::conditional_t<false, char, short>, short>::value);
  static_assert(std::is_same<
    std::decay_t<char const(&)[7]>, char const*>::value);
  static_assert(std::is_same<
    std::make_signed_t<unsigned char>, signed char>::value);
  static_assert(std::is_same<
    std::make_unsigned_t<signed char>, unsigned char>::value);
  static_assert(std::is_same<
    std::remove_all_extents_t<int[][10][10]>, int>::value);
  static_assert(std::is_same<
    std::remove_const_t<int const>, int>::value);
  static_assert(std::is_same<
    std::remove_cv_t<int const volatile>, int>::value);
  static_assert(std::is_same<
    std::remove_extent_t<int[]>, int>::value);
  static_assert(std::is_same<
    std::remove_pointer_t<int*>, int>::value);
  static_assert(std::is_same<
    std::remove_reference_t<int&>, int>::value);
  static_assert(std::is_same<
    std::remove_volatile_t<int volatile>, int>::value);
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
