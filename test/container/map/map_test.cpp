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
