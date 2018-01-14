#include <boost/intrusive/list.hpp>
#include <boost/intrusive/list_hook.hpp>

#include <iostream>

template <typename T>
class test_t : public boost::intrusive::list_base_hook<> {
public:
  test_t(T v) : v_(v) {}

  void set_val(T v) {
    v_ = v;
  }

  T get_val() {
    return v_;
  }
private:
  T v_;
};

auto main() -> decltype(0) {
  test_t<int>* a = new test_t(1);
  test_t b(2);
  test_t c = b;
  b.set_val(3);

  boost::intrusive::list<test_t<int> > l;

  l.push_back(*a);
  l.push_back(b);
  l.push_back(c);

  std::cout << a << '\n';
  std::cout << &l.front() << '\n';
  std::cout << &c << '\n';
  std::cout << &l.back() << '\n';
  std::cout << l.begin()->get_val() << '\n';
  std::cout << l.rbegin()->get_val() << '\n';

  return 0;
}
