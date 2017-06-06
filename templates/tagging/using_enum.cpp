#include <iostream>

template <typename T>
class do_swap {
private:
  void unsafe_swap(T& that) {
    std::cout << __func__ << '\n';
  }
public:
  void swap(T& that) {
    std::cout << __func__ << '\n';
  }

  enum swap_style {SAFE, UNSAFE};

  void swap(T& that, swap_style s) {
    if (s == SAFE)
      this->swap(that);
    else
      this->unsafe_swap(that);
  }
};

template <typename T>
T doit() {
  T t;
  do_swap<T> s;
  s.swap(t, s.SAFE);
}

auto main() -> int
{
  doit<int>();
  return 0;
}
