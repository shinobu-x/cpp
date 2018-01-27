#include <iostream>

#ifdef __cplusplus
  #undef offsetof
  #define offsetof(type, member)                                               \
      (reinterpret_cast<std::size_t>                                           \
        (&reinterpret_cast<const volatile char*&>                              \
          (static_cast<type*>(0)->member )                                     \
        )                                                                      \
      )
#endif

struct test_data {
  int a;
  char b[5];
  uint64_t c;
  long long int d;
};

auto main() -> decltype(0) {
  std::cout << offsetof(test_data, a) << '\n';
  std::cout << offsetof(test_data, b) << '\n';
  std::cout << offsetof(test_data, c) << '\n';
  std::cout << offsetof(test_data, d) << '\n';
  return 0;
}
