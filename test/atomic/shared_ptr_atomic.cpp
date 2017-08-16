#include <boost/shared_ptr.hpp>
#include <boost/memory_order.hpp>
#include "../macro/config.hpp"

struct dummy {};

auto main() -> decltype(0) {
  boost::shared_ptr<dummy> p1(new dummy);

  {
    boost::shared_ptr<dummy> p2 = boost::atomic_load(&p1);
    TEST_SP_EQ(p2, p1);

    boost::shared_ptr<dummy> p3(new dummy);
    boost::atomic_store(&p1, p3);
    TEST_SP_EQ(p1, p3);

    p2 = boost::atomic_load(&p1);
    TEST_SP_EQ(p2, p1);
    TEST_SP_EQ(p2, p3);

    boost::shared_ptr<dummy> p4(new dummy);
    boost::shared_ptr<dummy> p5 = boost::atomic_exchange(&p1, p4);
    TEST_SP_EQ(p5, p3);
    TEST_SP_EQ(p1, p4);

    boost::shared_ptr<dummy> p6(new dummy);
    boost::shared_ptr<dummy> cmp;
    bool r = boost::atomic_compare_exchange(&p1, &cmp, p6);
    assert(!r);
    TEST_SP_EQ(p1, p4);
    TEST_SP_EQ(cmp, p4);

    r = boost::atomic_compare_exchange(&p1, &cmp, p6);
    assert(r);
    TEST_SP_EQ(p1, p6);
  }

  p1.reset();

  {
    boost::shared_ptr<dummy> p2 = boost::atomic_load_explicit(&p1, 
      boost::memory_order_acquire);
    TEST_SP_EQ(p2, p1);

    boost::shared_ptr<dummy> p3(new dummy);
    boost::atomic_store_explicit(&p1, p3, boost::memory_order_release);
    TEST_SP_EQ(p1, p3);

    boost::shared_ptr<dummy> p4 = boost::atomic_exchange_explicit(&p1,
      boost::shared_ptr<dummy>(), boost::memory_order_acq_rel);
    TEST_SP_EQ(p4, p3);
    TEST_SP_EQ(p1, p2);

    boost::shared_ptr<dummy> p5(new dummy);
    boost::shared_ptr<dummy> cmp(p3);

    bool r = boost::atomic_compare_exchange_explicit(&p1, &cmp, p5,
      boost::memory_order_acquire, boost::memory_order_relaxed);
    assert(!r);
    TEST_SP_EQ(p1, p2);
    TEST_SP_EQ(cmp, p2);

    r = boost::atomic_compare_exchange_explicit(&p1, &cmp, p5,
      boost::memory_order_release, boost::memory_order_acquire);
    assert(r);
    TEST_SP_EQ(p1, p5);
  }

  return 0;
}
