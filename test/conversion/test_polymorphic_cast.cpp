#include <boost/polymorphic_cast.hpp>

#include <cassert>

namespace {
  struct Base1 {
    virtual char f1() { return '1'; }
  };

  struct Base2 {
    virtual char f2() { return '2'; }
  };

  struct Derived : public Base1, Base2 {
    virtual char f1() { return 'a'; }
  };
}  // namespace

auto main() -> decltype(0) {
  Derived derived1;
  Base1* base1 = &derived1;
  Derived* derived2 = boost::polymorphic_downcast<Derived*>(base1);
  assert(derived2->f1() == 'a');
  derived2 = boost::polymorphic_cast<Derived*>(base1);
  assert(derived2->f1() == 'a');
  Base2* base2 = boost::polymorphic_cast<Base2*>(base1);
  assert(base2->f2() == '2');
}
