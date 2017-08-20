#include <boost/container/detail/config_begin.hpp>
#include <boost/container/detail/pair.hpp>
#include <boost/move/iterator.hpp>
#include <boost/move/utility_core.hpp>
#include <boost/move/make_unique.hpp>
#include <boost/intrusive/detail/minimal_pair_header.hpp>
#include <boost/intrusive/detail/mpl.hpp>

#include <iostream>
#include <string>

#include "check_equal_containers.hpp"
#include "print_container.hpp"

namespace boost { namespace container { namespace test {
  BOOST_INTRUSIVE_HAS_MEMBER_FUNC_CALLED(has_member_rebalance, rebalance);
}}}

template <class T1, class T2, class T3, class T4>
bool operator==(std::pair<T1, T2>& p1, std::pair<T1, T2>& p2) {
  return p1.first == p2.first && p1.second == p2.second;
}

namespace {
  template <class C>
  void map_test_rebalanceable(C&,
    boost::container::container_detail::false_type) {}

  template <class C>
  void map_test_rebalanceable(C& c,
    boost::container::container_detail::true_type) {
    c.rebalance();
  }

  template <class boost_map, class std_map,
    class boost_multimap, class std_multimap>
  int map_test_copyable(boost::container::container_detail::false_type) {
    return 0;
  }

  template <class boost_map, class std_map,
    class boost_multimap, class std_multimap>
  int map_test_copyable(boost::container::container_detail::true_type) {
    typedef typename boost_map::key_type int_type;
    typedef boost::container::container_detail::pair<
      int_type, int_type> int_pair_type;
    typedef typename std_map::value_type std_pair_type;

    const int max_elem = 50;

    ::boost::movelib::unique_ptr<boost_map> const ptr_boost_map =
      ::boost::movelib::make_unique<boost_map>();

    ::boost::movelib::unique_ptr<std_map> const ptr_std_map =
      ::boost::movelib::make_unique<std_map>();

    ::boost::movelib::unique_ptr<boost_multimap> const ptr_boost_multimap =
      ::boost::movelib::make_unique<boost_multimap>();

    ::boost::movelib::unique_ptr<std_multimap> const ptr_std_multimap =
      ::boost::movelib::make_unique<std_multimap>();

    boost_map& boost_map_t = *ptr_boost_map;
    std_map& std_map_t = *ptr_std_map;
    boost_multimap& boost_multimap_t = *ptr_boost_multimap;
    std_multimap& std_multimap_t = *ptr_std_multimap;

    boost_map_t.insert(boost_map_t.cbegin(), boost_map_t.cend());
    boost_multimap_t.insert(
      boost_multimap_t.cbegin(), boost_multimap_t.cend());
    boost_map_t.insert(boost_map_t.begin(), boost_map_t.end());
    boost_multimap_t.insert(boost_multimap_t.begin(), boost_multimap_t.end());

    for (int i = 0; i < max_elem; ++i) {
      {
        int_type i1(i), i2(i);
        int_pair_type int_pair1(boost::move(i1), boost::move(i2));
        boost_map_t.insert(boost::move(int_pair1));
        std_map_t.insert(std_pair_type(i, i));
      }
      {
        int_type i1(i), i2(i);
        int_pair_type int_pair2(boost::move(i1), boost::move(i2));
        boost_multimap.insert(boost::move(int_pair2));
        std_multimap.insert(std_pair_type(i, i));
      }
    }

    if (!check_equal_containers(boost_map_t, std_map_t))
      return 1;

    if (!check_equal_containers(boost_multimap_t, std_multimap_t))
      return 1;

    {
      boost_map boost_map_copy(boost_map_t);
      std_map std_map_copy(std_map_t);
      boost_multimap boost_multimap_copy(boost_multimap_t);
      std_multimap std_multimap_copy(std_multimap_t);

      if (!check_equal_containers(boost_map_copy, std_map_copy))
        return 1;

      if (!check_equal_containers(boost_multimap_copy, std_multimap_copy))
        return 1;

      boost_map_copy = boost_map_t;
      std_map_copy = std_map_t;
      boost_multimap_copy = boost_multimap_t;
      std_multimap_copy = std_multimap_t;

      if (!check_equal_containers(boost_map_copy, std_map_copy))
        return 1;

      if (!check_equal_containers(boost_multimap_copy, std_multimap_copy))
        return 1;
    }

    return 0;
  }

  template <class boost_map, class std_map,
    class boost_multimap, class std_multimap>
  int map_test() {
    typedef typename boost_map::key_type int_type;
    typedef boost::container::container_detail::pair<
      int_type, int_type> int_pair_type;
    typedef typename std_map::value_type std_pair_type;
    const int max_elem = 50;
    typedef typename std_map::value_type std_value_type;
    typedef typename std_map::key_type std_key_type;
    typedef typename std_map::mapped_type std_mapped_type;

    {
      int_pair_type aux_vect1[max_elem];
      for (int i = 0; i < max_elem; ++i) {
        int_type i1(i/2);
        int_type i2(i/2);
        new(&aux_vect1[i])int_pair_type(boost::move(i1), boost::move(i2));
      }

      std_value_type aux_vect2[max_elem];
      for (int i = 0; i < max_elem; ++i)
        new(&aux_vect2[i])std_value_type(
          std_key_type(i/2), std_mapped_type(i/2));

      int_pair_type aux_vect3[max_elem];
      for (int i = 0; i < max_elem; ++i) {
        int_type i1(i/2);
        int_type i2(i/2);
        new(&aux_vect3[i])int_pair_type(boost::move(i1), boost::move(i2));
      }

      ::boost::movelib::unique_ptr<boost_map> const ptr_boost_map =
        ::boost::movelib::make_unique<boost_map>(
          boost::make_move_iterator(&aux_vect1[0]),
          boost::make_move_iterator(&aux_vect1[0] + max_elem),
          boost_map::key_compare());

      ::boost::movelib::unique_ptr<std_map> const ptr_std_map =
        ::boost::movelib::make_unique<std_map>(
          &aux_vect2[0],
          &aux_vect2[0] + max_elem,
          typename std_map::key_compare());

      if (!check_equal_containers(*ptr_boost_map, *ptr_std_map))
        return 1;

      ::boost::movelib::unique_ptr<boost_multimap> const ptr_boost_multimap =
        ::boost::movelib::make_unique<boost_multimap>(
          boost::make_move_iterator(&aux_vect3[0]),
          boost::make_move_iterator(&aux_vect3[0] + max_elem),
          typename boost_map::key_compare());

      ::boost::movelib::unique_ptr<std_multimap> const ptr_std_multimap =
        ::boost::movelib::make_unique<std_multimap>(
          &aux_vect2[0],
          &aux_vect2[0] + max_elem,
          typename std_map::key_compare());

      if (!check_equal_containers(*ptr_boost_multimap, *ptr_std_multimap))
        return 1;
    }
    {
      int_pair_type aux_vect1[max_elem];
      for (int i = 0 i < max_elem; ++i) {
        int_type i1(i/2);
        int_type i2(i/2);
        new(&aux_vect1[i])int_pair_type(boost::move(i1), boost::move(i2));
      }

      std_value_type aux_vect2[max_elem];
      for (int i = 0; i < max_elem; ++i)
        new(&aux_vect2[i])std_value_type(
          std_key_type(i/2), std_mapped_type(i/2));

      int_pair_type aux_vect3[max_elem];
      for (int i = 0; i < max_elem; ++i) {
        int_type i1(i/2);
        int_type i2(i/2);
        new(&aux_vect3[i])int_pair_type(boost::move(i1), boost::move(i2));
      }

      ::boost::movelib::unique_ptr<boost_map> const ptr_boost_map =
        ::boost::movelib::make_unique<boost_map>(
          boost::make_move_iterator(&aux_vect1[0]),
          boost::make_move_iterator(&aux_vect1[0] + max_elem),
          typename boost_map::allocator_type());

      ::boost::movelib::unique_ptr<std_map> const ptr_std_map =
        ::boost::movelib::make_unique<std_map>(
          &aux_vect2[0],
          &aux_vect2[0] + max_elem,
          typename std_map::key_compare());

      if (!check_equal_containers(*ptr_boost_map, *ptr_std_map0))
        return 1;

      ::boost::movelib::unique_ptr<boost_multimap> const ptr_boost_multimap =
        ::boost::movelib::make_unique<boost_multimap>(
          boost::make_move_iterator(&aux_vect3[0]),
          boost::make_move_iterator(&aux_vect3[0] + max_elem),
          typename boost_map::allocator_type());

      ::boost::movelib::unique_ptr<std_multimap> const ptr_std_multimap =
        ::boost::movelib::make_unique<std_multimap>(
          &aux_vect2[0],
          &aux_vect2[0] + max_elem,
          typename std_map::key_compare());

      if (!check_equal_containers(*ptr_boost_multimap, *ptr_std_multimap))
        return 1;
    }
    ::boost::movelib::unique_ptr<boost_map> const ptr_boost_map =
      ::boost::movelib::make_unique<boost_map>();
    ::boost::movelib::unique_ptr<std_map> const ptr_std_map =
      ::boost::movelib::make_unique<str_map>();
    ::boost::movelib::unique_ptr<boost_multimap> const ptr_boost_map =
      ::boost::movelib::make_unique<boost_multimap>();
    ::boost::movelib::unique_ptr<std_multimap> const ptr_std_multimap =
      ::boost::movelib::make_unique<std_multimap>();

    boost_map& boost_map_t = *ptr_boost_map;
    std_map& std_map_t = *ptr_std_map;
    boost_multimap& boost_multimap_t = *ptr_boost_multimap;
    std_multimap& std_multimap_t = *ptr_std_multimap;

    {
      int_pair_type aux_vect1[max_elem];
      for (int i = 0; i < max_elem; ++i) {
        int_type i1(i/2);
        int_type i2(i/2);
        new(&aux_vect1[i])int_pair_type(boost::move(i1), boost::move(i2));
      }

      typedef typename std_map::value_type std_value_type;
      typedef typename std_map::key_type std_key_type;
      typedef typename std_map::mapped_key std_mapped_type;
      std_value_type aux_vect2[max_elem];
      for (int i = 0; i < max_elem; ++i)
        new(&aux_vect2[i])std_value_type(boost::move(i1), boost::move(i2));

      int_pair_type aux_vect3[max_elem];
      for (int i = 0; i < max_elem; ++i) {
        int_type i1(i/2);
        int_type i2(i/2);
        new(&aux_vect3[i])int_pair_type(boost::move(i1), boost::move(i2));
      }

      ::boost::movelib::unique_ptr<boost_map> const ptr_boost_map2 =
        ::boost::movelib::make_unique<boost_map>(
          boost::make_move_iterator(&aux_vect1[0]),
          boost::make_move_iterator(&aux_vect1[0] + max_elem));

      ::boost::movelib::unique_ptr<std_map> const ptr_std_map2 =
        ::boost::movelib::make_unique<std_map>(
          &aux_vect2[0],
          &aux_vect2[0] + max_elem);

      ::boost::movelib::unique_ptr<boost_multimap> const ptr_boost_multimap2 =
        ::boost::movelib::make_unique<boost_multimap>(
          boost::make_move_iterator(&aux_vect3[0]),
          boost::make_move_iterator(&aux_vect3[0] + max_elem));

      ::boost::movelib::unique_ptr<std_multimap> const ptr_std_multimap2  =
        ::boost::movelib::make_unique<std_multimap>(
          &aux_vect2[0],
          &aux_vect2[0] + max_elem);

      boost_map& boost_map_t2 = *ptr_boost_map2;
      std_map& std_map_t2 = *ptr_std_map2;
      boost_multimap& boost_multimap_t2 = *ptr_boost_multimap2;
      std_multimap& std_multimap_t2 = *ptr_std_multimap2;

      if (!check_equal_containers(boost_map_t2, std_map_t2))
        return 1;

      if (!check_equal_containers(boost_multimap_t2, std_multimap_t2))
        return 1;

      for (int i = 0; i < max_elem; ++i) {
        int_type i1(i);
        int_type i2(i);
        new(&aux_vect1[i])int_pair_type(boost::move(i1), boost::move(i2));
      }

      for (int i = 0; i < max_elem; ++i) {
        new(&aux_vect2[i])std_value_type(std_key_type(i), std_mapped_type(i));

      for (int i = 0; i < max_elem; ++i) {
        int_type i1(i);
        int_type i2(i);
        new(&aux_vect3[i])int_pair_type(boost::move(i1), boost::move(i2));
      }

      if (!check_equal_containers(boost_map_t2, std_map_t2))
        return 1;

      if (!check_equal_containers(boost_multimap_t2, std_multimap_t2))
        return 1;

      if (!(boost_map_t2 == boost_map_t2))
        return 1;

      if (boost_map_t2 != boost_map_t2)
        return 1;

      if (boost_map_t2 < boost_map_t2)
        return 1;

      if (boost_map_t2 > boost_map_t2)
        return 1;

      if (!(boost_map_t2 <= boost_map_t2))
        return 1;

      if (!(boost_map_t2 >= boost_map_t2))
        return 1;

      ::boost::movelib::unique_ptr<boost_map> const ptr_boost_map3 =
        ::boost::movelib::make_unique<boost_map>(
          boost::make_move_iterator(&aux_vect1[0]),
          boost::make_move_iterator(&aux_vect1[0] + max_elem));

      ::boost::movelib::unique_ptr<std_map> const ptr_std_map3 =
        ::boost::movelib::make_unique<std_map>(
          &aux_vect2[0],
          &aux_vect2[0] + max_elem);

      ::boost::movelib::unique_ptr<boost_multimap> const ptr_boost_multimap3 =
        ::boost::movelib::make_unique<boost_multimap>(
          boost::make_move_iterator(&aux_vect3[0]),
          boost::make_move_iterator(&aux_vect3[0] + max_elem));

      ::boost::movelib::unique_ptr<std_multimap> const ptr_std_multimap3 =
        ::boost::movelib::make_unique<std_multimap>(
          &aux_vect2[0],
          &aux_vect2[0] + max_elem);

      boost_map& boost_map_t3 = *ptr_boost_map3;
      std_map& std_map_t3 = *ptr_std_map3;
      boost_multimap& boost_multimap_t3 = *ptr_boost_multimap3;
      std_multimap& std_multimap_t3 = *ptr_std_multimap3;

      if (!check_equal_containers(boost_map_t3, std_map_t3))
        return 1;

      if (!check_equal_containers(boost_multimap_t3, std_multimap_t3))
        return 1;

      {
        int_type i0(0);
        boost_map_t2.erase(i0);
        boost_multimap_t2.erase(i0);
        std_map_t2.erase(0);
        std_multimap_t2.erase(0);
      }
      {
        int_type i0(0);
        int_type i1(0);
        boost_map_t2[::boost::move(i0)] = ::boost::move(i1);
      }
      {
        int_type i1(1);
        boost_map_t2[int_type(0)] = ::boost::move(i1);
      }

      if (!check_equal_containers(boost_map_t2, std_map_t2))
        return 1;
    }
    {
      int_pair_type aux_vect1[max_elem];
      for (int i = 0; i < max_elem; ++i) {
        int_type i1(i);
        int_type i2(i);
        new(&aux_vect1[i])int_pair_type(boost::move(i1), boost::move(i2));
      }

      int_pair_type aux_vect2[max_elem];
      for (int i = 0; i < max_elem; ++i) {
        int_type i1(i);
        int_type i2(i);
        new(&aux_vect2[i])int_pair_type(boost::move(i1), boost::move(i2));
      }

      for (int i = 0; i < max_elem; ++i) {
        boost_map.insert(boost::move(aux_vect1[i]));
        std_map.insert(std_pair_type(i, i));
        boost_multimap.insert(boost::move(aux_vect2[i]));
        std_multimap.insert(std_pair_type(i, i));
      }  
} // namespace
