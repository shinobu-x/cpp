#include <iostream>

struct unsafe {};

template <typename T>
class do_swap
{
public:
  void swap(T& that){
    std::cout << "Safe" << '\n';
  };
  void swap(T& that, unsafe){
    std::cout << "Unsafe" << '\n';
  };
};

template <typename T>
T doit() {
  T t;
  unsafe u;
  do_swap<T> s;
  s.swap(t, u);
  s.swap(t);
}

auto main() -> int
{
  doit<int>();
  return 0;
}
