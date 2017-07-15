#include <algorithm>
#include <iostream>
#include <vector>

template <typename T>
struct type_t {
  type_t(std::vector<T> a) : a_(a) {};
  type_t(type_t const &that){ /** Your own logic **/ }
  ~type_t(){}

  void do_some() {
    if (!a_.empty())
//      std::cout << a_.size() << '\n';
      for_each (a_.begin(), a_.end(), [](T x) { std::cout << x << '\n'; });
  }

private:
  std::vector<T> a_;
};

template <typename T>
void doit() {
  std::vector<T> a;

  for (int i=0; i<10; ++i)
    a.push_back(i);

  type_t<T> b(a);

  b.do_some();

}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
