#include <boost/container/detail/config_begin.hpp>
#include <boost/container/map.hpp>
#include <boost/container/adaptive_pool.hpp>

#include <map>

// #include "print_container.hpp"
#include "movable_int.hpp"
#include "dummy_test_allocator.hpp"
#include "map_test.hpp"
#include "propagate_allocator_test.hpp"
#include "emplace_test.hpp"
#include "../../intrusive/iterator_test.hpp"

typedef std::pair<movable_and_copyable_int, movable_and_copyable_int> pair_t;

namespace boost { namespace container {
  template class map <
    movable_and_copyable_int,
    movable_and_copyable_int,
    std::less<movable_and_copyable_int>
    /* simple_allocator<pair_t>*/ >;


/*  template class map <
    movable_and_copyable_int,
    movable_and_copyable_int,
    std::less<movable_and_copyable_int>
    boost::container::adaptive_pool<pair_t> >; */

  template class multimap <
    movable_and_copyable_int,
    movable_and_copyable_int,
    std::less<movable_and_copyable_int>
    /* std::allocator<pair_t> */>;
}}

class recursive_map;
class recursive_multimap;
struct get_allocator_map;
struct boost_container_map;
struct boost_container_multimap;

class recursive_map {
public:
  recursive_map& operator=(const recursive_map& that) {
    id = that.id;
    map = that.map;
    return *this;
  }

  friend bool operator<(const recursive_map& a, const recursive_map& b) {
    return a.id < b.id;
  }

  int id;
  boost::container::map<recursive_map,
    recursive_map> map;
  boost::container::map<recursive_map,
    recursive_map>::iterator it;
  boost::container::map<recursive_map,
    recursive_map>::const_iterator cit;
  boost::container::map<recursive_map,
    recursive_map>::const_reverse_iterator crit;
};

class recursive_multimap {
public:
  recursive_multimap& operator=(const recursive_multimap& that) {
    id = that.id;
    multimap = that.multimap;
    return *this;
  }

  friend bool operator<(const recursive_multimap& a,
    const recursive_multimap& b) {
    return a.id < b.id;
  }

  int id;
  boost::container::multimap<recursive_multimap,
    recursive_multimap> multimap;
  boost::container::multimap<recursive_multimap,
    recursive_multimap>::iterator it;
  boost::container::multimap<recursive_multimap,
    recursive_multimap>::const_iterator cit;
  boost::container::multimap<recursive_multimap,
    recursive_multimap>::const_reverse_iterator crit;
};

template <class container_t>
void test_move() {
  container_t org;
  org.emplace();
  container_t move_ctor(boost::move(org));
  container_t move_assign;
  move_assign.emplace();
  move_assign = boost::move(move_ctor);
  move_assign.swap(org);
}

bool node_type_test() {
  {
    typedef boost::container::map<movable_int, movable_int> map_t;
    map_t src;

    {
      movable_int mvi1(1);
      movable_int mvi2(2);
      movable_int mvi3(3);
      movable_int mvi11(11);
      movable_int mvi12(12);
      movable_int mvi13(13);
      src.try_emplace(boost::move(mvi1), boost::move(mvi11));
      src.try_emplace(boost::move(mvi2), boost::move(mvi12));
      src.try_emplace(boost::move(mvi3), boost::move(mvi13));
    }

    if (src.size() != 3)
      return false;

    map_t dst;
    {
      movable_int mvi3(3);
      movable_int mvi33(33);
      dst.try_emplace(boost::move(mvi3), boost::move(mvi33));
    }
   
    if (dst.size() != 1)
      return false;

    const movable_int mvi1(1);
    const movable_int mvi2(2);
    const movable_int mvi3(3);
    const movable_int mvi33(33);
    const movable_int mvi13(13);

    map_t::insert_return_type r;

    r = dst.insert(src.extract(mvi33));
    if (!(r.position == dst.end() && r.inserted == false && r.node.empty()))
      return false;

    r = dst.insert(src.extract(src.find(mvi1)));
    if (!(r.position == dst.find(mvi1) && r.inserted == true && r.node.empty()))
      return false;

    r = dst.insert(dst.begin(), src.extract(mvi2));
    if (!(r.position == dst.find(mvi2) && r.inserted == true && r.node.empty()))
      return false;

    r = dst.insert(src.extract(mvi3));
    if (!src.empty())
      return false;
    if (dst.size() != 3)
      return false;
    if (!(r.position == dst.find(mvi3) && r.inserted == false &&
         r.node.key() == mvi3 && r.node.mapped() == mvi13))
      return false;
  }

  {
    typedef boost::container::multimap<movable_int, movable_int> multimap_t;
    multimap_t src;
    {
      movable_int mvi1(1);
      movable_int mvi2(2);
      movable_int mvi3(3);
      movable_int mvi3bits(3);
      movable_int mvi11(11);
      movable_int mvi12(12);
      movable_int mvi13(13);
      movable_int mvi23(23);

      src.emplace(boost::move(mvi1), boost::move(mvi11));
      src.emplace(boost::move(mvi2), boost::move(mvi12));
      src.emplace(boost::move(mvi3), boost::move(mvi13));
      src.emplace_hint(src.begin(), boost::move(mvi3bits), boost::move(mvi23));
    }

    if (src.size() != 4)
      return false;

    multimap_t dst;
    {
      movable_int mvi3(3);
      movable_int mvi33(33);
      dst.emplace(boost::move(mvi3), boost::move(mvi33));
    }

    if (dst.size() != 1)
      return false;

    const movable_int mvi1(1);
    const movable_int mvi2(2);
    const movable_int mvi3(3);
    const movable_int mvi4(4);
    const movable_int mvi33(33);
    const movable_int mvi13(13);
    const movable_int mvi23(23);

    multimap_t::iterator it;

    multimap_t::node_type nt(src.extract(mvi3));
    it = dst.insert(dst.begin(), boost::move(nt));
    if (!(it->first == mvi3 && it->second == mvi23 && dst.find(mvi3) == it &&
      nt.empty()))
      return false;

    nt = src.extract(src.find(mvi1));
    it = dst.insert(boost::move(nt));
    if (!(it->first == mvi1 && nt.empty()))
      return false;

    nt = src.extract(mvi2);
    it = dst.insert(boost::move(nt));
    if (!(it->first == mvi2 && nt.empty()))
      return false;

    it = dst.insert(src.extract(mvi3));
    if (!(it->first == mvi3 && it->second == mvi13 &&
      it == --multimap_t::iterator(dst.upper_bound(mvi3)) && nt.empty()))
      return false;

    it = dst.insert(src.extract(mvi4));
    if (!(it == dst.end()))
      return false;

    if (!src.empty())
      return false;

    if (dst.size() != 5)
      return false;
  }

  return true;

} // node_type_test

namespace {
template <>
struct alloc_propagate_base<boost_container_map>{
  template <class T, class allocator_t>
  struct apply {
    typedef typename boost::container::allocator_traits<allocator_t>::
      template portable_rebind_alloc<
        std::pair<const T, T> >::type alloc_t;

    typedef boost::container::multimap<T, T, std::less<T>, alloc_t> type;
  };
};
} // namespace

auto main() -> decltype(0) {
  return 0;
}
