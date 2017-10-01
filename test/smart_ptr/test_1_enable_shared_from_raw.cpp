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

struct C : public boost::enable_shared_from_raw {};

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

void test_2() {
  boost::shared_ptr<B> sptr_b(static_cast<impl2*>(0));
}

void test_3() {
  boost::shared_ptr<C> sptr_c(new C);

  try {
    boost::shared_ptr<C> sptr_c2 = boost::shared_from_raw(sptr_c.get());
    assert(sptr_c == sptr_c2);
    assert(!(sptr_c < sptr_c2) && !(sptr_c2 < sptr_c));
  } catch (boost::bad_weak_ptr const&) {}

  C c(*sptr_c);

  try {
    boost::shared_ptr<C> c2 = boost::shared_from_raw(&c);
    assert(c2.get() == &c);
    // Don't share onwership with sptr_c
    assert(sptr_c != c2);
    assert((sptr_c < c2) || (c2 < sptr_c));
  } catch (boost::bad_weak_ptr const&) {}

  try {
    *sptr_c = C();
    boost::shared_ptr<C> c3 = boost::shared_from_raw(sptr_c.get());
    assert(sptr_c == c3);
    assert(!(sptr_c < c3) && !(c3 < sptr_c));
  } catch (boost::bad_weak_ptr const&) {}
}

auto main() -> decltype(0) {
  test_1(); test_2(); test_3();
  return 0;
}
