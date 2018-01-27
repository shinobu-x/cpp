#include <iostream>

#ifdef __cplusplus
  #undef offsetof
  #define offsetof(type, member)                                               \
    (reinterpret_cast<std::size_t>                                             \
      (&reinterpret_cast<const volatile char*&>                                \
        (static_cast<type*>(0)->member)                                        \
      )                                                                        \
    )
#endif

template <typename T>
struct alignmentof {
private:
  struct alignment {
    char _a;
    T _b;
  };
public:
  std::size_t value = offsetof(alignment, _b);
};

struct test_data {
  int a;
  char b[5];
  double c;
};

auto main() -> decltype(0) {
  alignmentof<test_data> align;
  std::cout << align.value << '\n';
  return 0;
}
