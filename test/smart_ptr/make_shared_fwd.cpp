#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include <cassert>

class args {
public:
  args() {}
private:
  args(args&& other) {}
  args& operator= (args&& other) { return *this; }
  args(const args& other) {}
  args& operator= (const args& other) { return *this; }
};

class A {
public:
  enum constructor_id {
    move,
    const_reference,
    reference
  };

  A(args&& arg) : constructed_by_(move) {}
  A(const args& arg) : constructed_by_(const_reference) {}
  A(args& arg) : constructed_by_(reference) {}

  constructor_id constructed_by_;
};

struct B {
  B(int& value) : ref(value) {}
  int& ref;
};

auto main() -> decltype(0) {
  {
    args a;
    boost::shared_ptr<A> b = boost::make_shared<A>(a);
    assert(b->constructed_by_ == A::reference);
  }
  {
    const args ca;
    boost::shared_ptr<A> b = boost::make_shared<A>(ca);
    assert(b->constructed_by_ == A::const_reference);
  }
  {
    boost::shared_ptr<A> a = boost::make_shared<A>(args());
    assert(a->constructed_by_ == A::move);
  }
  {
    int value = 1;
    boost::shared_ptr<B> a = boost::make_shared<B>(value);
    assert(a->ref == 1 && value == a->ref);
    ++a->ref;
    assert(value == a->ref);
  }
  return 0;
}
