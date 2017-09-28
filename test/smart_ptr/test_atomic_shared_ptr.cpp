#include <boost/shared_ptr.hpp>
#include <boost/memory_order.hpp>

#include <cassert>

struct dummy {};

auto main() -> decltype(0) {

  boost::shared_ptr<dummy> sptr1(new dummy);
  
  {
    boost::shared_ptr<dummy> sptr2 = boost::atomic_load(&sptr1);
    assert(sptr2 == sptr1);

    boost::shared_ptr<dummy> sptr3(new dummy);
    boost::atomic_store(&sptr1, sptr3);
    assert(sptr1 == sptr3);

    sptr2 = boost::atomic_load(&sptr1);
    assert(sptr2 == sptr1);
    assert(sptr2 == sptr3);

    boost::shared_ptr<dummy> sptr4(new dummy);
    boost::shared_ptr<dummy> sptr5 = boost::atomic_exchange(&sptr1, sptr4);
    assert(sptr3 == sptr5);
    assert(sptr1 == sptr4);

    boost::shared_ptr<dummy> sptr6(new dummy);
    boost::shared_ptr<dummy> cmp;

    bool r = boost::atomic_compare_exchange(&sptr1, &cmp, sptr6);
    assert(!r);
    assert(sptr1 == sptr4);
    assert(cmp == sptr4);

    r = boost::atomic_compare_exchange(&sptr1, &cmp, sptr6);
    assert(r);
    assert(sptr1 == sptr6);
  }

  return 0;
}
