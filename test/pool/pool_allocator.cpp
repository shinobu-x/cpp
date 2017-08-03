#include <boost/pool/pool_alloc.hpp>
#include <boost/pool/object_pool.hpp>

#include <algorithm>
#include <deque>
#include <list>
#include <set>
#include <stdexcept>
#include <vector>

#include <cassert>
#include <cstdlib> /* size_t */
#include <ctime>

class checker {
public:
  bool ok() const { return objs.empty(); }

  ~checker() { /* assert(ok()); */ }

  void in(void* const this_obj) {
    /* assert(this_obj.find(this_obj) == objs_.end()); */
    objs.insert(this_obj);
  }

  void out(void* const this_obj) {
    /* assert(objs_.find(this_obj) != objs_.end()); */
    objs.erase(this_obj);
  }
private:
  std::set<void*> objs;
};

static checker ck;

struct tester {
  tester(bool throw_except = false) {
    if (throw_except)
      throw std::logic_error("LE");

    ck.in(this);
  }

  tester(const tester&) {
    ck.in(this);
  }

  ~tester() {
    ck.out(this);
  }
};

template <typename Allocator>
struct tracker {
  typedef typename Allocator::size_type size_type;
  typedef typename Allocator::difference_type difference_type;

  static std::set<char*> allocated_blocks;

  static char* malloc(const size_type bytes) {
    char* const r = Allocator::malloc(bytes);
    allocated_blocks.insert(r);
    return r;
  }

  static void free(char* const block) {
    assert(allocated_blocks.find(block) != allocated_blocks.end());
    allocated_blocks.erase(block);
    Allocator::free(block);
  }

  static bool ok() {
    return allocated_blocks.empty();
  }
};

template <typename Allocator>
std::set<char*> tracker<Allocator>::allocated_blocks;

typedef tracker<boost::default_user_allocator_new_delete> tracker_t;

void test_1() {
  {
    boost::object_pool<tester> pool;
  }

  {
    boost::object_pool<tester> pool;
    for (unsigned i = 0; i < 10; ++i)
      pool.construct();
  }

  {
    boost::object_pool<tester> pool;
    std::vector<tester*> v;
    for (unsigned i = 0; i < 10; ++i)
      v.push_back(pool.construct());
  }

  {
    boost::object_pool<tester> pool;
    std::vector<tester*> v;
    for (unsigned i = 0; i < 10; ++i)
      v.push_back(pool.construct());
    std::random_shuffle(v.begin(), v.end());
    for (unsigned j = 0; j < 10; ++j)
      pool.destroy(v[j]);
  }

  {
    boost::object_pool<tester> pool;
    for (unsigned i = 0; i < 5; ++i)
      pool.construct();
    for (unsigned j = 0; j < 5; ++j)
      try {
        pool.construct(true);
      } catch (const std::logic_error&) {}
  }
}

void test_2() {
  {
    std::vector<tester, boost::pool_allocator<tester> > v;
    for (unsigned i = 0; i < 10; ++i)
      v.push_back(tester());
    v.pop_back();
  }

  {
    std::deque<tester, boost::pool_allocator<tester> > v;
    for (unsigned i = 0; i < 10; ++i)
      v.push_back(tester());
    v.pop_back();
    v.pop_front();
  }

  {
    std::list<tester, boost::fast_pool_allocator<tester> > v;
    for (unsigned i = 0; i < 10; ++i)
      v.push_back(tester());
    v.pop_back();
    v.pop_front();
  }

  tester* tmp;
  {
    boost::pool_allocator<tester> a;
    tmp = a.allocate(1, 0);
    new (tmp)tester();
  }

/*  if (ck.ok()) */

  tmp->~tester();
  boost::pool_allocator<tester>::deallocate(tmp, 1);

  {
    boost::pool_allocator<tester> alloc;
    tester* tp = alloc.allocate(0);
    alloc.deallocate(tp, 0);
  }
}

void test_3() {
  typedef boost::pool<tracker_t> pool_t;

  {
    pool_t pool(sizeof(int));
    assert(tracker_t::ok());
    assert(!pool.release_memory());
    assert(!pool.purge_memory());

    pool.free(pool.malloc());
    assert(!tracker_t::ok());

    assert(pool.release_memory());
    assert(tracker_t::ok());

    pool.malloc();

    assert(!pool.release_memory());

    assert(pool.purge_memory());
    assert(tracker_t::ok());

    pool.malloc();
  } 
}

void test_4() {
  typedef boost::pool_allocator<void> void_allocator;
  typedef boost::fast_pool_allocator<void> fast_void_allocator;

  typedef void_allocator::rebind<int>::other int_allocator;
  typedef fast_void_allocator::rebind<int>::other fast_int_allocator;

  std::vector<int, int_allocator> v1;
  std::vector<int, fast_int_allocator> v2;
}

auto main() -> decltype(0) {
  test_1(); test_2(); test_3();
  return 0;
}
