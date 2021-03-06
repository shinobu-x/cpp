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
        boost_multimap_t.insert(boost::move(int_pair2));
        std_multimap_t.insert(std_pair_type(i, i));
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
          typename boost_map::key_compare());

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
          typename boost_map::allocator_type());

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
      ::boost::movelib::make_unique<std_map>();
    ::boost::movelib::unique_ptr<boost_multimap> const ptr_boost_multimap =
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
      typedef typename std_map::mapped_type std_mapped_type;
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

      for (int i = 0; i < max_elem; ++i) 
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
        boost_map_t.insert(boost::move(aux_vect1[i]));
        std_map_t.insert(std_pair_type(i, i));
        boost_multimap_t.insert(boost::move(aux_vect2[i]));
        std_multimap_t.insert(std_pair_type(i, i));
      }

      if (!check_equal_pair_containers(boost_map_t, std_map_t))
        return 1;

      if (!check_equal_pair_containers(boost_multimap_t, std_multimap_t))
        return 1;

      typename boost_map::iterator it = boost_map_t.begin();
      typename boost_map::const_iterator cit = it;
      (void)cit;

      boost_map_t.erase(boost_map_t.begin());
      std_map_t.erase(std_map_t.begin());
      boost_multimap_t.erase(boost_multimap_t.begin());
      std_multimap_t.erase(std_multimap_t.begin());

      if (!check_equal_pair_containers(boost_map_t, std_map_t))
        return 1;

      if (!check_equal_pair_containers(boost_multimap_t, std_multimap_t))
        return 1;

      boost_map_t.erase(boost_map_t.begin());
      std_map_t.erase(std_map_t.begin());
      boost_multimap_t.erase(boost_multimap_t.begin());
      std_multimap_t.erase(std_multimap_t.begin());

      if (!check_equal_pair_containers(boost_map_t, std_map_t))
        return 1;

      if (!check_equal_pair_containers(boost_multimap_t, std_multimap_t))
        return 1;

      boost_map tmp_boost_map1;
      std_map tmp_std_map1;
      boost_multimap tmp_boost_multimap1;
      std_multimap tmp_std_multimap1;
      boost_map_t.swap(tmp_boost_map1);
      std_map_t.swap(tmp_std_map1);
      boost_multimap_t.swap(tmp_boost_multimap1);
      std_multimap_t.swap(tmp_std_multimap1);
      boost_map_t.swap(tmp_boost_map1);
      std_map_t.swap(tmp_std_map1);
      boost_multimap_t.swap(tmp_boost_multimap1);
      std_multimap_t.swap(tmp_std_multimap1);

      if (!check_equal_pair_containers(boost_map_t, std_map_t))
        return 1;

      if (!check_equal_pair_containers(boost_multimap_t, std_multimap_t))
        return 1;
    }
    {
      int_pair_type aux_vect1[max_elem];
      for (int i = 0; i < max_elem; ++i) {
        int_type i1(-1);
        int_type i2(-1);
        new(&aux_vect1[i])int_pair_type(boost::move(i1), boost::move(i2));
      }
      int_pair_type aux_vect2[max_elem];
      for (int i = 0; i < max_elem; ++i) {
        int_type i1(-1);
        int_type i2(-1);
        new(&aux_vect2[i])int_pair_type(boost::move(i1), boost::move(i2));
      }

      boost_map_t.insert(
        boost::make_move_iterator(&aux_vect1[0]),
        boost::make_move_iterator(&aux_vect1[0] + max_elem));

      boost_multimap_t.insert(
        boost::make_move_iterator(&aux_vect2[0]),
        boost::make_move_iterator(&aux_vect2[0] + max_elem));

      for (int i = 0; i != max_elem; ++i) {
        std_pair_type std_pair_type(-1, -1);
        std_map_t.insert(std_pair_type);
        std_multimap_t.insert(std_pair_type);
      }

      if (!check_equal_pair_containers(boost_map_t, std_map_t))
        return 1;

      if (!check_equal_pair_containers(boost_multimap_t, std_multimap_t))
        return 1;
    }
    {
      int_pair_type aux_vect1[max_elem];
      for (int i = 0; i < max_elem; ++i) {
        int_type i1(-1);
        int_type i2(-1);
        new(&aux_vect1[i])int_pair_type(boost::move(i1), boost::move(i2));
      }

      int_pair_type aux_vect2[max_elem];
      for (int i = 0; i < max_elem; ++i) {
        int_type i1(-1);
        int_type i2(-1);
        new(&aux_vect2[i])int_pair_type(boost::move(i1), boost::move(i2));
      }

      int_pair_type aux_vect3[max_elem];
      for (int i = 0; i < max_elem; ++i) {
        int_type i1(-1);
        int_type i2(-1);
        new(&aux_vect3[i])int_pair_type(boost::move(i1), boost::move(i2));
      }

      int_pair_type aux_vect4[max_elem];
      for (int i = 0; i < max_elem; ++i) {
        int_type i1(-1);
        int_type i2(-1);
        new(&aux_vect4[i])int_pair_type(boost::move(i1), boost::move(i2));
      }

      boost_map_t.insert(
        boost::make_move_iterator(&aux_vect1[0]),
        boost::make_move_iterator(&aux_vect1[0] + max_elem));
      boost_map_t.insert(
        boost::make_move_iterator(&aux_vect2[0]),
        boost::make_move_iterator(&aux_vect2[0] + max_elem));
      boost_multimap_t.insert(
        boost::make_move_iterator(&aux_vect3[0]),
        boost::make_move_iterator(&aux_vect3[0] + max_elem));
      boost_multimap_t.insert(
        boost::make_move_iterator(&aux_vect4[0]),
        boost::make_move_iterator(&aux_vect4[0] + max_elem));

      for (int i = 0; i != max_elem; ++i) {
        std_pair_type std_pair_type(-1, -1);
        std_map_t.insert(std_pair_type);
        std_multimap_t.insert(std_pair_type);
        std_map_t.insert(std_pair_type);
        std_multimap_t.insert(std_pair_type);
      }

      boost_map_t.erase(boost_map_t.begin()->first);
      std_map_t.erase(std_map_t.begin()->first);
      boost_multimap_t.erase(boost_multimap_t.begin()->first);
      std_multimap_t.erase(std_multimap_t.begin()->first);

      if (!check_equal_pair_containers(boost_map_t, std_map_t))
        return 1;

      if (!check_equal_pair_containers(boost_multimap_t, std_multimap_t))
        return 1;
    }
    {
      int_pair_type aux_vect1[max_elem];
      for (int i = 0; i < max_elem; ++i) {
        int_type i1(i);
        int_type i2(i);
        new(&aux_vect1[i])int_pair_type(
          boost::move(i1), boost::move(i2));
      }

      int_pair_type aux_vect2[max_elem];
      for (int i = 0; i < max_elem; ++i) {
        int_type i1(i);
        int_type i2(i);
        new(&aux_vect1[i])int_pair_type(
          boost::move(i1), boost::move(i2));
      }

      for (int i = 1; i < max_elem; ++i) {
        boost_map_t.insert(boost::move(aux_vect1[i]));
        std_map_t.insert(std_pair_type(i, i));
        boost_multimap_t.insert(boost::move(aux_vect2[i]));
        std_multimap_t.insert(std_pair_type(i, i));
      }

      if (!check_equal_pair_containers(boost_map_t, std_map_t))
        return 1;

      if (!check_equal_pair_containers(boost_multimap_t, std_multimap_t))
        return 1;

      for (int i = 0; i < max_elem; ++i) {
        int_pair_type int_pair;
        {
          int_type i1(i);
          int_type i2(i);
          new(&int_pair)int_pair_type(boost::move(i1), boost::move(i2));
        }
        boost_map_t.insert(boost_map_t.begin(), boost::move(int_pair));
        std_map_t.insert(std_multimap_t.begin(), std_pair_type(i, i));

        {
          int_type i1(i);
          int_type i2(i);
          new(&int_pair)int_pair_type(boost::move(i1), boost::move(i2));
        }
        boost_multimap_t.insert(
          boost_multimap_t.begin(), boost::move(int_pair));
        std_multimap_t.insert(std_multimap_t.begin(), std_pair_type(i, i));

        if (!check_equal_pair_containers(boost_map_t, std_map_t))
          return 1;

        if (!check_equal_pair_containers(boost_multimap_t, std_multimap_t))
          return 1;

        {
          int_type i1(i);
          int_type i2(i);
          new(&int_pair)int_pair_type(boost::move(i1), boost::move(i2));
        }
        boost_map_t.insert(boost_map_t.end(), boost::move(int_pair));
        std_map_t.insert(std_map_t.end(), std_pair_type(i, i));

        {
          int_type i1(i);
          int_type i2(i);
          new(&int_pair)int_pair_type(boost::move(i1), boost::move(i2));
        }
        boost_multimap_t.insert(boost_multimap_t.end(), boost::move(int_pair));
        std_map_t.insert(std_map_t.end(), std_pair_type(i, i));

        if (!check_equal_pair_containers(boost_map_t, std_map_t))
          return 1;

        if (!check_equal_pair_containers(boost_multimap_t, std_multimap_t))
          return 1;

        {
          std::pair<typename boost_multimap::iterator,
            typename boost_multimap::iterator> br = 
              boost_multimap_t.equal_range(boost_multimap_t.begin()->first);

          std::pair<typename std_multimap::iterator,
            typename std_multimap::iterator> sr =
              std_multimap_t.equal_range((std_multimap_t.begin()->first));

          if (boost::container::iterator_distance(br.first, br.second) !=
            boost::container::iterator_distance(sr.first, sr.second))
            return 1;
        }
        {
          int_type i1(i);
          boost_map_t.insert(boost_map_t.upper_bound(
            boost::move(i1)), boost::move(int_pair));
          std_map_t.insert(std_map_t.upper_bound(i), std_pair_type(i, i));
        }
        {
          int_type i1(i);
          int_type i2(i);
          new(&int_pair)int_pair_type(boost::move(i1), boost::move(i2));
        }
        {
          int_type i1(i);
          boost_multimap_t.insert(boost_multimap_t.upper_bound(
            boost::move(i1)), boost::move(int_pair));
          std_multimap_t.insert(std_multimap_t.upper_bound(i),
            std_pair_type(i, i));
        }

        if (!check_equal_pair_containers(boost_map_t, std_map_t))
          return 1;

        if (!check_equal_pair_containers(boost_multimap_t, std_multimap_t))
          return 1;

        map_test_rebalanceable(boost_map_t,
          boost::container::container_detail::bool_<
            boost::container::test::has_member_rebalance<
              boost_map>::value>());

        if (!check_equal_containers(boost_map_t, std_map_t))
          return 1;

        map_test_rebalanceable(boost_multimap_t,
          boost::container::container_detail::bool_<
            boost::container::test::has_member_rebalance<
              boost_map>::value>());

        if (!check_equal_containers(boost_multimap_t, std_multimap_t))
          return 1;

      }

      for (int i = 0; i < max_elem; ++i) {
        int_type k(i);
        if (boost_map_t.count(k) != std_map_t.count(i))
          return -1;

        if (boost_multimap_t.count(k) != std_multimap_t.count(i))
          return -1;
      }

      boost_map_t.erase(boost_map_t.begin(), boost_map_t.end());
      boost_multimap_t.erase(boost_multimap_t.begin(), boost_multimap_t.end());
      boost_map_t.clear();
      boost_multimap_t.clear();

      for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 100; ++i) {
          int_pair_type int_pair;
          {
            int_type i1(i);
            int_type i2(i);
            new(&int_pair)int_pair_type(boost::move(i1), boost::move(i2));
          }
          boost_map_t.insert(boost::move(int_pair));
          {
            int_type i1(i);
            int_type i2(i);
            new(&int_pair)int_pair_type(boost::move(i1), boost::move(i2));
          }
          boost_multimap_t.insert(boost::move(int_pair));

          int_type k(i);

          if (boost_map_t.count(k) != typename boost_map::size_type(1))
            return 1;

          if (boost_multimap_t.count(k) !=
            typename boost_multimap::size_type(j + 1))
            return 1;
      }
    }

    {
      boost_map_t.clear();
      std_map_t.clear();
      boost_multimap_t.clear();
      std_multimap_t.clear();

      int_pair_type aux_vect1[max_elem];
      for (int i = 0; i < max_elem; ++i) {
        int_type i1(i);
        int_type i2(i);
        new(&aux_vect1[i])int_pair_type(boost::move(i1), boost::move(i2));
      }

      for (int i = 0; i < max_elem; ++i) {
        boost_map_t[boost::move(aux_vect1[i].first)] =
          boost::move(aux_vect1[i].second);
        std_map_t[i] = i;
      }

      if (!check_equal_pair_containers(boost_map_t, std_map_t))
        return 1;

      if (!check_equal_pair_containers(boost_multimap_t, std_multimap_t))
        return 1;
    }


    {
      boost_map_t.clear();
      std_map_t.clear();
      boost_multimap_t.clear();
      std_multimap_t.clear();

      int_pair_type aux_vect1[max_elem];
      for (int i = 0; i < max_elem; ++i) {
        int_type i1(i);
        int_type i2(i);
        new(&aux_vect1[i])int_pair_type(boost::move(i1), boost::move(i2));
      }

      int_pair_type aux_vect2[max_elem];
      for (int i = 0; i < max_elem; ++i) {
        int_type i1(i);
        int_type i2(max_elem - 1);
        new(&aux_vect2[i])int_pair_type(boost::move(i1), boost::move(i2));
      }

      for (int i = 0; i < max_elem; ++i) {
        boost_map_t.insert_or_assign(boost::move(aux_vect1[i].first),
          boost::move(aux_vect1[i].second));
        std_map_t[i] = i;
      }

      if (!check_equal_pair_containers(boost_map_t, std_map_t))
        return 1;

      if (!check_equal_pair_containers(boost_multimap_t, std_multimap_t))
        return 1;

      for (int i = 0; i < max_elem; ++i) {
        boost_map_t.insert_or_assign(boost::move(aux_vect2[i].first),
          boost::move(aux_vect2[i].second));
        std_map_t[i] = max_elem - 1;
      }

      if (!check_equal_pair_containers(boost_map_t, std_map_t))
        return 1;

      if (!check_equal_pair_containers(boost_multimap_t, std_multimap_t))
        return 1;
    }


    {
      boost_map_t.clear();
      std_map_t.clear();
      boost_multimap_t.clear();
      std_multimap_t.clear();

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

      typedef typename boost_map::iterator iterator_t;
      for (int i = 0; i < max_elem; ++i) {
        iterator_t it;

        if (i&1) {
          std::pair<typename boost_map::iterator, bool> r =
            boost_map_t.try_emplace(boost::move(aux_vect1[i].first),
              boost::move(aux_vect1[i].second));

          if (!r.second)
            return 1;
          it = r.first;
        } else
          it = boost_map_t.try_emplace(
            boost_map_t.upper_bound(aux_vect1[i].first),
            boost::move(aux_vect1[i].first),
            boost::move(aux_vect1[i].second));

       if (boost_map_t.end() == it || 
         it->first != aux_vect2[i].first || it->second != aux_vect2[i].first)
         return 1;

       std_map_t[i] = i;
      }

      if (!check_equal_pair_containers(boost_map_t, std_map_t))
        return 1;

      if (!check_equal_pair_containers(boost_multimap_t, std_multimap_t))
        return 1;

      for (int i = 0; i < max_elem; ++i) {
        iterator_t it;
        iterator_t itex = boost_map_t.find(aux_vect2[i].first);

        if (i&1) {
          std::pair<typename boost_map::iterator, bool> r =
            boost_map_t.try_emplace(boost::move(aux_vect2[i].first),
              boost::move(aux_vect2[i].second));

          if (r.second)
            return 1;

          it = r.first;
        } else
          it = boost_map_t.try_emplace(
            boost_map_t.upper_bound(aux_vect2[i].first),
            boost::move(aux_vect2[i].first),
            boost::move(aux_vect2[i].second));

        const int_type test_int(i);

        if (boost_map_t.end() == it || it != itex || it->second != test_int)
          return 1;
      }

      if (!check_equal_pair_containers(boost_map_t, std_map_t))
        return 1;

      if (!check_equal_pair_containers(boost_multimap_t, std_multimap_t))
        return 1;
    }


    {
      ::boost::movelib::unique_ptr<boost_map> const ptr_boost_map2 =
        ::boost::movelib::make_unique<boost_map>();
      ::boost::movelib::unique_ptr<boost_multimap> const ptr_boost_multimap2 =
        ::boost::movelib::make_unique<boost_multimap>();

      boost_map& boost_map2 = *ptr_boost_map2;
      boost_multimap& boost_multimap2 = *ptr_boost_multimap2;

      boost_map_t.clear();
      boost_map2.clear();
      boost_multimap_t.clear();
      boost_multimap2.clear();
      std_map_t.clear();
      std_multimap_t.clear();

      {
        int_pair_type aux_vect1[max_elem];
        for (int i = 0; i < max_elem; ++i) {
          int_type i1(i);
          int_type i2(i);
          new(&aux_vect1[i])int_pair_type(boost::move(i1), boost::move(i2));
        }

        int_pair_type aux_vect2[max_elem];
        for (int i = 0; i < max_elem; ++i) {
          int_type i1(max_elem/2+i);
          int_type i2(max_elem-i);
          new(&aux_vect2[i])int_pair_type(boost::move(i1), boost::move(i2));
        }

        int_pair_type aux_vect3[max_elem];
        for (int i = 0; i < max_elem; ++i) {
          int_type i1(max_elem*2/2+i);
          int_type i2(max_elem*2+i);
          new(&aux_vect3[i])int_pair_type(boost::move(i1), boost::move(i2));
        }

        boost_map_t.insert(
          boost::make_move_iterator(&aux_vect1[0]),
          boost::make_move_iterator(&aux_vect1[0] + max_elem));

        boost_map2.insert(
          boost::make_move_iterator(&aux_vect2[0]),
          boost::make_move_iterator(&aux_vect2[0] + max_elem));

        boost_multimap2.insert(
          boost::make_move_iterator(&aux_vect3[0]),
          boost::make_move_iterator(&aux_vect3[0] + max_elem));
      }

      for (int i = 0; i < max_elem; ++i)
        std_map_t.insert(std_pair_type(i, i));

      for (int i = 0; i < max_elem; ++i)
        std_map_t.insert(std_pair_type(max_elem/2+i, max_elem-i));

      boost_map_t.merge(boost::move(boost_map2));

      if (!check_equal_pair_containers(boost_map_t, std_map_t))
        return 1;

      for (int i = 0; i < max_elem; ++i)
        std_map_t.insert(std_pair_type(max_elem*2/2+i, max_elem*2+i));

      boost_map_t.merge(boost::move(boost_multimap2));

      if (!check_equal_pair_containers(boost_map_t, std_map_t))
        return 1;

      boost_map2.clear();
      boost_multimap_t.clear();
      boost_multimap2.clear();
      std_map_t.clear();
      std_multimap_t.clear();

      {
        int_pair_type aux_vect1[max_elem];
        for (int i = 0; i < max_elem; ++i) {
          int_type i1(i);
          int_type i2(i);
          new(&aux_vect1[i])int_pair_type(boost::move(i1), boost::move(i1));
        }

        int_pair_type aux_vect2[max_elem];
        for (int i = 0; i < max_elem; ++i) {
          int_type i1(max_elem/2+i);
          int_type i2(max_elem-i);
          new(&aux_vect2[i])int_pair_type(boost::move(i1), boost::move(i2));
        }

        int_pair_type aux_vect3[max_elem];
        for (int i = 0; i < max_elem; ++i) {
            int_type i1(max_elem*2/2+i);
            int_type i2(max_elem*2+i);
            new(&aux_vect3[i])int_pair_type(boost::move(i1), boost::move(i2));
        }

        boost_multimap_t.insert(
          boost::make_move_iterator(&aux_vect1[0]),
          boost::make_move_iterator(&aux_vect1[0] + max_elem));

        boost_multimap2.insert(
          boost::make_move_iterator(&aux_vect2[0]),
          boost::make_move_iterator(&aux_vect2[0] + max_elem));

        boost_map2.insert(
          boost::make_move_iterator(&aux_vect3[0]),
          boost::make_move_iterator(&aux_vect3[0] + max_elem));
      }

      for (int i = 0; i < max_elem; ++i)
        std_multimap_t.insert(std_pair_type(i, i));

      for (int i = 0; i < max_elem; ++i)
        std_multimap_t.insert(std_pair_type(max_elem/2+i, max_elem-i));

      boost_multimap_t.merge(boost::move(boost_multimap2));

      if (!check_equal_pair_containers(boost_multimap_t, std_multimap_t))
        return 1;

      for (int i = 0; i < max_elem; ++i)
        std_multimap_t.insert(std_pair_type(max_elem*2/2+i, max_elem*2+i));

      boost_multimap_t.merge(boost::move(boost_multimap2));

      if (!check_equal_pair_containers(boost_multimap_t, std_multimap_t))
        return 1;
      
    }

    if (map_test_copyable<boost_map, std_map, boost_multimap, std_multimap>(
      boost::container::container_detail::bool_<
        is_copyable<int_type>::value>()))
      return 1;

    return 0;
  }
  template <typename map_type>
  bool test_map_support_for_initialization_list_for() {
#if !defined(BOOST_NO_CX11_HDR_INITIALIZER_LIST)
    const std::initializer_list<
      std::pair<typename map_type::value_type::first_type,
      typename map_type::mapped_type> > il =
      {
        std::make_pair(1, 2),
        std::make_pair(3, 4)
      };

    const map_type expected_map(il.begin(), il.end());
    {
      const map_type il1 = il;

      if (il1 != expected_map)
        return false;

      map_type il2(il, typename map_type::key_compare(),
        typename map_type::allocator_type());

      if (il2 != expected_map)
        return false;

      const map_type il_ordered(boost::container::ordered_unique_range, il);

      if (il_ordered != expected_map)
        return false;

      map_type il_assign = { std::make_pair(99, 100) };
      il_assign = il;

      if (il_assign != expected_map)
        return false;
    }
    {
      map_type il1;
      il1.insert(il);
      if (il1 != expected_map)
        return false;
    }

    return true;
#endif
    return true;
  }

  template <typename map_type, typename multimap_type>
  bool instantiate_constructors() {
    {
      typedef typename map_type::value_type value_type;
      typename map_type::key_compare comp;
      typename map_type::allocator_type alloc;
      value_type value;

      {
        map_type m1;
        map_type m2(comp);
        map_type m3(alloc);
        map_type m4(comp, alloc);
      }
      {
        map_type m1(&value, &value);
        map_type m2(&value, &value, comp);
        map_type m3(&value, &value, alloc);
        map_type m4(&value, &value, comp, alloc);
      }
#if !defined(BOOST_NO_CXX11_HDR_INITIALIZER_LIST)
      {
        std::initializer_list<value_type> il;
        map_type m1(il);
        map_type m2(il, comp);
        map_type m3(il, alloc);
        map_type m4(il, comp, alloc);
      }
      {
        std::initializer_list<value_type> il;
        map_type m1(boost::container::ordered_unique_range, il);
        map_type m2(boost::container::ordered_unique_range, il, comp);
        map_type m3(boost::container::ordered_unique_range, il, comp, alloc);
      }
#endif
      {
        map_type m1(boost::container::ordered_unique_range, &value, &value);
        map_type m2(
          boost::container::ordered_unique_range, &value, &value, comp);
        map_type m3(
          boost::container::ordered_unique_range, &value, &value, comp, alloc);
      }
    }

    {
      typedef typename multimap_type::value_type value_type;
      typename multimap_type::key_compare comp;
      typename multimap_type::allocator_type alloc;
      value_type value;

      {
        multimap_type m1;
        multimap_type m2(comp);
        multimap_type m3(alloc);
        multimap_type m4(comp, alloc);
      }
      {
        multimap_type m1(&value, &value);
        multimap_type m2(&value, &value, comp);
        multimap_type m3(&value, &value, alloc);
        multimap_type m4(&value, &value, comp, alloc);
      }
#if !defined(BOOST_NO_CXX11_HDR_INITIALIZER_LIST)
      {
        std::initializer_list<value_type> il;
        multimap_type m1(il);
        multimap_type m2(il, comp);
        multimap_type m3(il, alloc);
        multimap_type m4(il, comp, alloc);
      }
      {
        std::initializer_list<value_type> il;
        multimap_type m1(boost::container::ordered_range, il);
        multimap_type m2(boost::container::ordered_range, il, comp);
        multimap_type m3(boost::container::ordered_range, il, comp, alloc);
      }
#endif
      {
        multimap_type m1(boost::container::ordered_range, &value, &value);
        multimap_type m2(boost::container::ordered_range, &value, &value, comp);
        multimap_type m3(
          boost::container::ordered_range, &value, &value, comp, alloc);
      }
    }

    return true;
  }
} // namespace
