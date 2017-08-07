#include <boost/smart_ptr/enable_shared_from_raw.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>

#include <cassert>
#include <memory>
#include <string>

#include "../macro/config.hpp"

class A : public boost::enable_shared_from_raw {
private:
  int destroyed_;
  int deleted_;
  int expected_;

  A(A const&);
  A& operator= (A const&);

public:
  static int instances;

  explicit A(int expected)
    : destroyed_(0), deleted_(0), expected_(expected) {
    ++instances;
  }

  ~A() {
    assert(deleted_ == expected_);
    assert(destroyed_ == 0);
    ++destroyed_;
    --instances;
  }

  typedef void (*deleter_type)(A*);

  static void deleter(A* pa) {
    ++pa->deleted_;
  }

  static void deleter2(A* pa) {
    ++pa->deleted_;
    delete pa;
  }
};

int A::instances = 0;

struct TEST1 : public boost::enable_shared_from_raw {
  virtual ~TEST1() {}
  std::string m1_;
};

struct TEST2 {
  virtual ~TEST2() {}
  std::string m2_;
};

struct TEST3 : TEST2, TEST1 {};

void test_1() {
  assert(A::instances == 0);

  {
    A a(0);
    assert(A::instances == 1);
  }
  assert(A::instances == 0);
  {
    std::auto_ptr<A> pa(new A(0));
    assert(A::instances == 1);
  }
  assert(A::instances == 0);
  {
    boost::shared_ptr<A> spa(new A(0));
    assert(A::instances == 1);
    boost::weak_ptr<A> wpa(spa);
    assert(!wpa.expired());
    spa.reset();
    assert(wpa.expired());
  }
  assert(A::instances == 0);
  {
    A a(1);
    boost::shared_ptr<A> pa(&a, A::deleter);
    assert(A::instances == 1);
    A::deleter_type* pad = boost::get_deleter<A::deleter_type>(pa);
    assert(pad != 0 && *pad == A::deleter);
    boost::weak_ptr<A> wpa(pa);
    assert(!wpa.expired());
    pa.reset();
    assert(wpa.expired());
  }
  assert(A::instances == 0);
  {
    boost::shared_ptr<A> pa(new A(1), A::deleter2);
    assert(A::instances == 1);
    A::deleter_type* pad = boost::get_deleter<A::deleter_type>(pa);
    assert(pad != 0 && *pad == A::deleter2);
    boost::weak_ptr<A> wpa(pa);
    assert(!wpa.expired());
    pa.reset();
    assert(wpa.expired());
  }
  assert(A::instances == 0);
}

void test_2() {
  boost::shared_ptr<TEST3> p(new TEST3);
}

void test_3() {
  TEST1* p1 = new TEST3;
  boost::shared_ptr<void> p2(p1);
  assert(p2.get() == p1);
  assert(p2.use_count() == 1);
}

struct null_deleter {
  void operator() (void const*) const {}
};

void test_4() {
  boost::shared_ptr<TEST1> p1_1(new TEST1);
  boost::shared_ptr<TEST1> p1_2(p1_1.get(), null_deleter());
  assert(p1_2.get() == p1_1.get());
  assert(p1_2.use_count() == 1);
}

void test_5() {
  TEST1 p1_1;
  boost::shared_ptr<TEST1> p1_2(&p1_1, null_deleter());
  assert(p1_2.get() == &p1_1);
  assert(p1_2.use_count() == 1);

  try {
    boost::shared_from_raw(p1_2.get());
  } catch (...) {
LOG_ERROR;
  }

//  p1_2.reset();

  boost::shared_ptr<TEST1> p1_3(&p1_1, null_deleter());
  assert(p1_3.get() == &p1_1);
  assert(p1_3.use_count() == 1);

  try {
    boost::shared_from_raw(p1_2.get());
  } catch (...) {
LOG_ERROR;
  }
}

auto main() -> decltype(0) {
  test_1(); test_2(); test_3();  test_4(); test_5();
  return 0;
}
