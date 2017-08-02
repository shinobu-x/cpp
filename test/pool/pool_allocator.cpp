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
    objs_insert(this_obj);
  }

  void out(void* const this_obj) {
    /* assert(objs_.find(this_obj) != objs_.end()); */
    objs_.erase(this_obj);
  }
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
    sk.out(this);
  }
};

template <typename Allocator>
struct tracker {
  typedef typename Allocator::size_type size_type;
  typedef typename Allocator::difference_type diff_type;

  static std::set<char*> allocated_blocks;

  static char* malloc(const size_type bytes) {
    char* const r = Allocator::malloc(bytes);
    allocated_blocks.insert(r);
    return r;
  }



private:
  std::set<void*> objs_;
};
