#include <iostream>

template <typename T, int N>
class recursive {
  T s;
  recursive<T, N - 1> ra_;
public:
  T size() const {
    return 1 + ra_.size();
  }
};

template <typename T>
class recursive<T, 0> {
public:
  T size() const {
    return 0;
  }
};

template <typename T>
T doit() {
  recursive<T, 5> a;
  T r = a.size();
  std::cout << r << '\n';
}

auto main() -> int
{
  doit<int>();
  return 0;
}
