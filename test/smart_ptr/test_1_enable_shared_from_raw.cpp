#include <boost/smart_ptr/enable_shared_from_raw.hpp>

#include <boost/shared_ptr.hpp>

#include <cassert>

class A {
public:
  virtual void f() = 0;
protected:
  ~A() {}
};

class B {
public:
  virtual boost::shared_ptr<A> getA() = 0;
protected:
  ~B() {}
};

class impl1
  : public A, public virtual B, public virtual boost::enable_shared_from_raw {
public:
  virtual void f() {}

  virtual boost::shared_ptr<A> getA() {
    boost::shared_ptr<impl1> sptr_this = boost::shared_from_raw(this);
    assert(sptr_this.get() == this);
    return sptr_this;
  }
};

class impl2 : public impl1 {};

boost::shared_ptr<B> createB() {
  boost::shared_ptr<B> sptr_b(new impl2);
  return sptr_b;
}

void test_1() {
  boost::shared_ptr<B> sptr_b = createB();
  assert(sptr_b.get() != 0);
  assert(sptr_b.use_count() == 1);
  try {
    boost::shared_ptr<A> sptr_a = sptr_b->getA();
    assert(sptr_a.get() != 0);
    assert(sptr_b.use_count() == 2);
    sptr_a->f();
#if !defined(BOOST_NO_RTTI)
    boost::shared_ptr<B> sptr_b2 = boost::dynamic_pointer_cast<B>(sptr_a);
    assert(sptr_b.get() == sptr_b2.get());
    assert(!(sptr_b < sptr_b2 || sptr_b2 < sptr_b));
    assert(sptr_b.use_count() == 3);
#endif
  } catch (boost::bad_weak_ptr const&) {}
}

auto main() -> decltype(0) {
  test_1();
  return 0;
}
