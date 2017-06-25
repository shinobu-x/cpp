#include <iostream>

class incomplete_type;
class complete_type {};

template <size_t N>
struct compile_time_const {
  complete_type& operator== (compile_time_const<N>) const;

  template <size_t K>
  incomplete_type& operator== (compile_time_const<K>) const;
};

template <typename T>
compile_time_const<0> length_of(T) {
  return compile_time_const<0>();
}

template <typename T, size_t N>
compile_time_const<N> length_of(T (&)[N]) {
  return compile_time_const<N>();
}

template <typename T>
T doit() {
  T a[10];
}

auto main() -> int
{
  doit<int>();
  return 0;
}
