#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

class A {
public:
  A() {}
  A(const char* n) : n_(n) {}
  const std::string& get_n() const { return n_; }
  void print_n(std::ostream& os) const { os << n_ << " "; }
  void set_n(const std::string& n) { n_ = n; std::cout << n_ << " "; }
  std::string clear_n() { std::string r = n_; n_ = ""; return r; }
  void do_some(int) const {}
  bool is_what() const { return n_ == "what"; }
private:
  std::string n_;
};

namespace {
  bool is_equal(const std::string& s1, const std::string& s2) {
    return s1 == s2;
  }

  bool is_who(const std::string& s) {
    return s == "who";
  }

  void do_set_n_ptr(A* p, const std::string& n) {
    p->set_n(n);
  }

  void do_set_n_ref(A& r, const std::string& n) {
    r.set_n(n);
  }
} // namespace

auto main() -> decltype(0) {
  std::vector<A> v1;
  v1.push_back("what");
  v1.push_back("who");
  v1.push_back("where");
  v1.push_back("when");
  v1.push_back("how");

  const std::vector<A> cv1(v1.begin(), v1.end());

  std::vector<A> v2;
  v2.push_back("who");
  v2.push_back("where");
  v2.push_back("when");
  v2.push_back("how");
  v2.push_back("what");

  A a;
  A& ar = a;

  A what("what");
  A who("who");
  A where("where");
  A when("when");
  A how("how");
  std::vector<A*> v3;
  v3.push_back(&what);
  v3.push_back(&who);
  v3.push_back(&where);
  v3.push_back(&when);
  v3.push_back(&how);

  const std::vector<A*> cv3(v3.begin(), v3.end());
  std::vector<const A*> vc3(v3.begin(), v3.end());

  std::ostream& os = std::cout;

//  std::cout << '\n';
  std::transform(v3.begin(), v3.end(),
    std::ostream_iterator<std::string>(std::cout, " "),
    std::mem_fun(&A::get_n));

  std::cout << '\n';
  std::transform(cv3.begin(), cv3.end(),
    std::ostream_iterator<std::string>(std::cout, " "),
    std::mem_fun(&A::get_n));

  std::cout << '\n';
  std::transform(vc3.begin(), vc3.end(),
    std::ostream_iterator<std::string>(std::cout, " "),
    std::mem_fun(&A::get_n));

  std::cout << '\n';
  std::transform(v1.begin(), v1.end(),
    std::ostream_iterator<std::string>(std::cout, " "),
    std::mem_fun_ref(&A::get_n));

  std::cout << '\n';
  std::transform(cv1.begin(), cv1.end(),
    std::ostream_iterator<std::string>(std::cout, " "),
    std::mem_fun_ref(&A::get_n));

  std::cout << '\n';
  std::transform(v3.begin(), v3.end(),
    std::ostream_iterator<std::string>(std::cout, " "),
    std::mem_fun(&A::clear_n));

  std::cout << '\n';
  std::transform(v1.begin(), v1.end(),
    std::ostream_iterator<std::string>(std::cout, " "),
    std::mem_fun_ref(&A::clear_n));

  std::cout << '\n';

  return 0;
}
