#include <boost/container/detail/config_begin.hpp>
#include <boost/container/detail/pair.hpp>
#include <boost/container/detail/mpl.hpp>
#include <boost/move/unique_ptr.hpp>

#include <cstddef>
#include <boost/container/detail/iterator.hpp>

namespace {
  template <class T1, class T2>
  bool check_equal(const T1 &t1, const T2 &t2,
    typename boost::container::container_detail::enable_if_c<
        !boost::container::container_detail::is_pair<T1>::value &&
        !boost::container::container_detail::is_pair<T2>::value
      >::type* = 0) {
    return t1 == t2;
  }

  template <class T1, class T2, class C1, class C2>
  bool check_equal_it(const T1& i1, const T2& i2, const C1& c1, const C2& c2) {
    bool c1_end = i1 == c1.end();
    bool c2_end = i2 == c2.end();

    if (c1_end != c2_end)
      return false;
    else if (c1_end)
      return true;
    else
      return check_equal(*i1, *i2);
  }

  template <class Pair1, class Pair2>
  bool check_equal(const Pair1& pair1, const Pair2& pair2,
    typename boost::container::container_detail::enable_if_c<
        boost::container::container_detail::is_pair<Pair1>::value &&
        boost::container::container_detail::is_pair<Pair2>::value
      >::type* = 0) {
    return check_equal(pair1.first, pair2.first) &&
      check_equal(pair1.second, pair2.second);
  }

  template <class C1, class C2>
  bool check_equal_containers(const C1& c1, const C2& c2) {
    if (c1.size() != c2.size())
      return false;

    typename C1::const_iterator i1(c1.begin()), i1_end(c1.end());
    typename C2::const_iterator i2(c2.begin()), i2_end(c2.end());
    typename C2::size_type dist1 =
      (typename C2::size_type)boost::container::iterator_distance(i1, i1_end);

    if (dist1 != c1.size())
      return false;

    typename C1::size_type dist2 =
      (typename C1::size_type)boost::container::iterator_distance(i2, i2_end);

    if (dist2 != c2.size())
      return false;

    std::size_t i = 0;

    for (; i1 != i1_end; ++i1, ++i2, ++i)
      if (!check_equal(*i1, *i2))
        return false;

    return true;
  }

  template <class boost_type, class std_type>
  bool check_equal_pair_containers(const boost_type &boost_t,
    const std_type std_t) {
    if (boost_t.size() != std_t.size())
      return false;

    typedef typename boost_type::key_type key_type;
    typedef typename std_type::mapped_type mapped_type;

    typename boost_type::const_iterator boost_it(boost_t.begin()),
      boost_it_end(boost_t.end());
    typename std_type::const_iterator std_it(std_t.begin());

    for (; boost_it != boost_it_end; ++boost_it, ++std_it) {
      key_type k(std_it->first);

      if (boost_it->first != k)
        return false;

      mapped_type m(std_it->second);

      if (boost_it->second != m)
        return false;

    }

    return true;
}
} // namespace
