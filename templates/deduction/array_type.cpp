#include <iostream>

template <typename T, std::size_t N>
constexpr std::size_t array_size(T (&)[N]) noexcept {
  return N;  /// N will be used at compile time.
}

template <typename T>
void doit() {
  T a[] = {1, 2, 3, 4, 5};
  std::cout << array_size(a) << '\n';
}
auto main() -> int
{
  doit<int>();
  return 0;
}
