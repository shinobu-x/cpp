#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>

#include <cassert>

struct empty : public boost::enable_shared_from_this<empty> {};

struct dummy {
  dummy();
};

static dummy obj;

static boost::shared_ptr<empty> sptr1;
static boost::weak_ptr<empty> wptr1;
static boost::shared_ptr<empty> sptr2(nullptr);

dummy::dummy() {
  sptr1.reset(new empty);
  wptr1 = sptr1;
  sptr2.reset(new empty);
}

auto main() -> decltype(0) {
  assert(sptr1.get() != 0);
  assert(sptr1.use_count() == 1);
  assert(wptr1.use_count() == 1);
  assert(wptr1.lock() == sptr1);
  assert(sptr2.get() != 0);
  assert(sptr2.use_count() == 1);  
  return 0;
}
