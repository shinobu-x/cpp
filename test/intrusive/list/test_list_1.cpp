#include <boost/intrusive/list.hpp>

#include <iostream>
#include <vector>

class test_t : public boost::intrusive::list_base_hook<> {
public:
  friend bool operator==(const test_t& lhs, const test_t& rhs) {
    return lhs.v_ == rhs.v_;
  }

  int v_;
};

typedef boost::intrusive::list<test_t> test_l;

struct clone_t {
  test_t* operator()(const test_t& other) {
    return new test_t(other);
  }
};

struct deleter {
  void operator()(test_t* other) {
    delete other;
  }
};

auto main() -> decltype(0) {
  std::vector<test_t> v(100);
  test_l list;
  for (int i = 0; i < 100; ++i)
    v[i].v_ = i;

  list.insert(list.end(), v.begin(), v.end());

  test_l clone;

  clone.clone_from(list, clone_t(), deleter());

  if (clone != list)
    std::cout << "NG\n";
  else
    std::cout << "OK\n";

  clone.clear_and_dispose(deleter());

  return 0;
}
