#include <boost/intrusive/detail/iterator.hpp>
#include <boost/intrusive/detail/mpl.hpp>
#include <boost/static_assert.hpp>

template <class container_type1>
struct has_member_reverse_iterator {
  typedef char yes_type;
  struct no_type {
    char _[2];
  };

  template <class container_type2>
  static no_type test(...);

  template <typename container_type2>
  static yes_type test(typename container_type2::reverse_iterator const*);

  static const bool value =
    sizeof(test<container_type1>(0)) == sizeof(yes_type);
};

template <class container_type1>
struct has_member_const_reverse_iterator {
  typedef char yes_type;
  struct no_type {
    char _[2];
  };

  template <class container_type2>
  static no_type test(...);

  template <class container_type2>
  static yes_type test(
    typename container_type2::const_reverse_iterator const*);

  static const bool value =
    sizeof(test<container_type1>(0)) == sizeof(yes_type);
};

template <class container_type,
  bool = has_member_reverse_iterator<container_type>::value>
struct get_reverse_iterator {
  typedef typename container_type::reverse_iterator type;

  static type begin(container_type& c) {
    return c.rbegin();
  }

  static type end(container_type& c) {
    return c.rend();
  }
};

template <class container_type>
struct get_reverse_iterator<container_type, false> {
  typedef typename container_type::iterator type;

  static type begin(container_type& c) {
    return c.begin();
  }

  static type end(container_type& c) {
    return c.end();
  }
};

template <class container_type,
  bool = has_member_const_reverse_iterator<container_type>::value>
struct get_const_reverse_iterator {
  typedef typename container_type::const_reverse_iterator type;

  static type begin(container_type& c) {
    return c.crbegin();
  }

  static type end(container_type& c) {
    return c.crend();
  }
};

template <class container_type>
struct get_const_reverse_iterator<container_type, false> {
  typedef typename container_type::const_iterator type;

  static type begin(container_type& c) {
    return c.cbegin();
  }

  static type end(container_type& c) {
    return c.end();
  }
};

template <class iterator_type>
void test_iterator_operations(iterator_type begin, iterator_type end) {
  BOOST_TEST(begin != end);
  BOOST_TEST(!(begin == end));

  {
    iterator_type it1;
    iterator_type it2(begin); /* Copy constructible */
    it1 = it2; /* Assignable */
    (void)it1; /* Destructible */
    (void)it2;
  }

  typedef typename boost::intrusive::iterator_traits<
    iterator_type>::reference reference;
  reference r = *begin;
  (void)r;

  typedef typename boost::intrusive::iterator_traits<
    iterator_type>::pointer pointer;
  pointer p = (boost::intrusive::iterator_arrow_result)(begin);
  (void)p;

  iterator_type& rit = ++begin;
  (void)rit;

  const iterator_type& crit = begin++;
  (void)crit;
}

template <class container_type>
void test_iterator_compatible(container_type& c) {
  typedef container_type type;
  typedef typename type::iterator iterator;
  typedef typename type::const_iterator const_iterator;
  typedef typename get_reverse_iterator<type>::type reverse_iterator;
  typedef typename get_const_reverse_iterator<type>::type
    const_reverse_iterator;

  test_iterator_operations(c.begin(), c.end());
  test_iterator_operations(c.cbegin(), c.cend());
  test_iterator_operations(
    get_reverse_iterator<type>::begin(c),
    get_reverse_iterator<type>::end(c));
  test_iterator_operations(
    get_const_reverse_iterator<type>::begin(c),
    get_const_reverse_iterator<type>::end(c));

  BOOST_STATIC_ASSERT(
    (!boost::container::container_detail::is_convertible<
      const_iterator, iterator>::value));

  BOOST_STATIC_ASSERT(
    (!boost::container::container_detail::is_convertible<
      const_reverse_iterator, reverse_iterator>::value));

  {
    const_iterator cit1;
    iterator it(c.begin());
    cit1 = it;
    (void)cit1;
    BOOST_ASSERT(cit1 == it);
    BOOST_ASSERT(*cit1 == *it);

    const_iterator cit2(it);
    BOOST_ASSERT(cit2 == it);
    BOOST_ASSERT(*cit2 == *it);
  }

  {
    const_reverse_iterator crit1;
    reverse_iterator rit(get_reverse_iterator<type>::begin(c));
    crit1 = rit;
    BOOST_ASSERT(crit1 == rit);
    BOOST_ASSERT(*crit1 == *rit);

    const_reverse_iterator crit2(rit);
    BOOST_ASSERT(crit2 == rit);
    BOOST_ASSERT(*crit2 == *rit);
    (void)crit2;
  }
}

template <class container_type>
void test_iterator_input_and_compatible(container_type& c) {
  typedef container_type type;
  typedef typename type::iterator iterator;
  typedef typename type::const_iterator const_iterator;
  typedef typename get_reverse_iterator<type>::type reverse_iterator;
  typedef typename get_const_reverse_iterator<type>::type
    const_reverse_iterator;

  typedef boost::intrusive::iterator_traits<iterator> it_traits;
  typedef boost::intrusive::iterator_traits<const_iterator> cit_traits;
  typedef boost::intrusive::iterator_traits<reverse_iterator> rit_traits;
  typedef boost::intrusive::iterator_traits<
    const_reverse_iterator> crit_traits;

  BOOST_STATIC_ASSERT(
    (!boost::move_detail::is_same<iterator, const_iterator>::value));
  BOOST_STATIC_ASSERT(
    (!boost::move_detail::is_same<
      reverse_iterator, const_reverse_iterator>::value));

  /* difference_type */
  typedef typename type::difference_type difference_type;

  BOOST_STATIC_ASSERT(
    (boost::move_detail::is_same<
      difference_type, typename it_traits::difference_type>::value));
  BOOST_STATIC_ASSERT(
    (boost::move_detail::is_same<
      difference_type, typename cit_traits::difference_type>::value));
  BOOST_STATIC_ASSERT(
    (boost::move_detail::is_same<
      difference_type, typename rit_traits::difference_type>::value));
  BOOST_STATIC_ASSERT(
    (boost::move_detail::is_same<
      difference_type, typename crit_traits::difference_type>::value));

  /* value_type */
  typedef typename type::value_type value_type;
  BOOST_STATIC_ASSERT(
    (boost::move_detail::is_same<
      value_type, typename it_traits::value_type>::value));
  BOOST_STATIC_ASSERT(
    (boost::move_detail::is_same<
      value_type, typename cit_traits::value_type>::value));
  BOOST_STATIC_ASSERT(
    (boost::move_detail::is_same<
      value_type, typename rit_traits::value_type>::value));
  BOOST_STATIC_ASSERT(
    (boost::move_detail::is_same<
      value_type, typename crit_traits::value_type>::value));

  /* pointer type */
  typedef typename type::pointer pointer;
  typedef typename type::const_pointer const_pointer;
  BOOST_STATIC_ASSERT(
    (boost::move_detail::is_same<
      pointer, typename it_traits::pointer>::value));
  BOOST_STATIC_ASSERT(
    (boost::move_detail::is_same<
      const_pointer, typename cit_traits::pointer>::value));
  BOOST_STATIC_ASSERT(
    (boost::move_detail::is_same<
      pointer, typename rit_traits::pointer>::value));
  BOOST_STATIC_ASSERT(
    (boost::move_detail::is_same<
     const_pointer, typename crit_traits::pointer>::value));

  /* reference type */
  typedef typename type::reference reference;
  typedef typename type::const_reference const_reference;
  BOOST_STATIC_ASSERT(
    (boost::move_detail::is_same<
      reference, typename it_traits::reference>::value));
  BOOST_STATIC_ASSERT(
    (boost::move_detail::is_same<
      const_reference, typename cit_traits::reference>::value));
  BOOST_STATIC_ASSERT(
    (boost::move_detail::is_same<
      reference, typename rit_traits::reference>::value));
  BOOST_STATIC_ASSERT(
    (boost::move_detail::is_same<
      const_reference, typename crit_traits::reference>::value));

  test_iterator_compatible(c);
}

template <class container_type, class iterator_type>
void test_iterator_forward_functions(container_type& c,
  iterator_type const begin, iterator_type const end) {
  typedef typename container_type::size_type size_type;

  {
    size_type i = 0;
    iterator_type it1 = begin;
    for (iterator_type it2 = begin; i != c.size(); ++it1, ++i) {
      BOOST_TEST(it1 == it2++);
      iterator_type it_tmp(it1);
      iterator_type* it_addr = &it_tmp;
      BOOST_TEST(&(++it_tmp) == it_addr);
      BOOST_TEST(it_tmp == it2);
    }

    BOOST_TEST(i == c.size());
    BOOST_TEST(it1 == end);
  }
}

template <class container_type>
void test_iterator_forward_and_compatible(container_type& c) {
  test_iterator_input_and_compatible(c);
  test_iterator_forward_functions(c, c.begin(), c.end());
  test_iterator_forward_functions(c, c.cbegin(), c.cend());
  test_iterator_forward_functions(c,
    get_reverse_iterator<container_type>::begin(c),
    get_reverse_iterator<container_type>::end(c));
  test_iterator_forward_functions(c,
    get_const_reverse_iterator<container_type>::begin(c),
    get_const_reverse_iterator<container_type>::end(c));
}

template <class container_type, class iterator_type>
void test_iterator_bidirectional_functions(container_type& c,
  iterator_type const begin, iterator_type const end) {
  typedef typename container_type::size_type size_type;

  {
    size_type i = 0;
    iterator_type it1 = end;
    for (iterator_type it2 = end; i != c.size(); --it1, ++i) {
      BOOST_TEST(it1 == it2--);
      iterator_type it_tmp(it1);
      iterator_type* it_addr = &it_tmp;
      BOOST_TEST(&(--it_tmp) == it_addr);
      BOOST_TEST(it_tmp == it2);
      BOOST_TEST((++it_tmp) == it1);
    }

    BOOST_TEST(i == c.size());
    BOOST_TEST(it1 == begin);
  }
}

template <class container_type>
void test_iterator_bidirectional_and_compatible(container_type& c) {
  test_iterator_forward_and_compatible(c);
  test_iterator_bidirectional_functions(c, c.begin(), c.end());
  test_iterator_bidirectional_functions(c, c.cbegin(), c.cend());
  test_iterator_bidirectional_functions(c, c.rbegin(), c.rend());
  test_iterator_bidirectional_functions(c, c.crbegin(), c.crend());
}

template <class container_type, class iterator_type>
void test_iterator_random_functions(container_type& c,
  iterator_type const begin, iterator_type const end) {
  typedef typename container_type::size_type size_type;


  {
    iterator_type it = begin;

    for (size_type i = 0, m = c.size(); i != m; ++i, ++it) {
      BOOST_TEST(i == size_type(it - begin));
      BOOST_TEST(begin[i] == *it);
      BOOST_TEST(&begin[i] == &*it);
      BOOST_TEST((begin + i) == it);
      BOOST_TEST((i + begin) == it);
      BOOST_TEST(begin == (it - i));

      iterator_type tmp(begin);
      BOOST_TEST((tmp += i) == it);
      tmp = it;
      BOOST_TEST(c.size() == size_type(end - begin));
    }
  }

  {
    iterator_type it1(begin);
    iterator_type it2(begin);

    if (begin != end) {
      for (++it1; it1 != end; ++it1) {
        BOOST_TEST(it2 < it1);
        BOOST_TEST(it2 <= it1);
        BOOST_TEST(!(it2 > it1));
        BOOST_TEST(!(it2 >= it1));
        BOOST_TEST(it1 > it2);
        BOOST_TEST(it1 >= it2);
        BOOST_TEST(!(it1 < it2));
        BOOST_TEST(!(it1 <= it2));
        BOOST_TEST(it1 >= it1);
        BOOST_TEST(it1 <= it1);
        it2 = it1;
      }
    }
  }
}

template <class container_type>
void test_iterator_random_and_compatible(container_type& c) {
  test_iterator_bidirectional_and_compatible(c);
  test_iterator_random_functions(c, c.begin(), c.end());
  test_iterator_random_functions(c, c.cbegin(), c.cend());
  test_iterator_random_functions(c, c.rbegin(), c.rend());
  test_iterator_random_functions(c, c.crbegin(), c.crend());
}

template <class container_type>
void test_iterator_forward(container_type& c) {
  typedef container_type type;
  typedef typename type::iterator iterator;
  typedef typename type::const_iterator const_iterator;
  typedef typename get_reverse_iterator<type>::type reverse_iterator;
  typedef typename get_const_reverse_iterator<type>::type
    const_reverse_iterator;

  typedef boost::intrusive::iterator_traits<iterator> it_traits;
  typedef boost::intrusive::iterator_traits<const_iterator> cit_traits;
  typedef boost::intrusive::iterator_traits<reverse_iterator> rit_traits;
  typedef boost::intrusive::iterator_traits<
    const_reverse_iterator> crit_traits;

  /* iterator_category */
  BOOST_STATIC_ASSERT(
    (boost::container::container_detail::is_same<std::forward_iterator_tag,
      typename it_traits::iterator_category>::value));
  BOOST_STATIC_ASSERT(
    (boost::container::container_detail::is_same<std::forward_iterator_tag,
      typename cit_traits::iterator_category>::value));
  BOOST_STATIC_ASSERT(
    (boost::container::container_detail::is_same<std::forward_iterator_tag,
      typename rit_traits::iterator_category>::value));
  BOOST_STATIC_ASSERT(
    (boost::container::container_detail::is_same<std::forward_iterator_tag,
      typename crit_traits::iterator_category>::value));

  test_iterator_forward_and_compatible(c);
}

template <class container_type>
void test_iterator_bidirectional(container_type& c) {
  typedef container_type type;
  typedef typename type::iterator iterator;
  typedef typename type::const_iterator const_iterator;
  typedef typename type::reverse_iterator reverse_iterator;
  typedef typename type::const_reverse_iterator const_reverse_iterator;
  typedef boost::intrusive::iterator_traits<iterator> it_traits;
  typedef boost::intrusive::iterator_traits<const_iterator> cit_traits;
  typedef boost::intrusive::iterator_traits<reverse_iterator> rit_traits;
  typedef boost::intrusive::iterator_traits<
    const_reverse_iterator> crit_traits;

  /* iterator_category */
  BOOST_STATIC_ASSERT(
    (boost::container::container_detail::is_same<
      std::bidirectional_iterator_tag,
      typename it_traits::iterator_category>::value));
  BOOST_STATIC_ASSERT(
    (boost::container::container_detail::is_same<
      std::bidirectional_iterator_tag,
      typename cit_traits::iterator_category>::value));
  BOOST_STATIC_ASSERT(
    (boost::container::container_detail::is_same<
      std::bidirectional_iterator_tag,
      typename rit_traits::iterator_category>::value));
  BOOST_STATIC_ASSERT(
    (boost::container::container_detail::is_same<
      std::bidirectional_iterator_tag,
      typename crit_traits::iterator_category>::value));

  test_iterator_bidirectional_and_compatible(c);;
}

template <class container_type>
void test_iterator_random(container_type& c) {
  typedef container_type type;
  typedef typename type::iterator iterator;
  typedef typename type::const_iterator const_iterator;
  typedef typename type::reverse_iterator reverse_iterator;
  typedef typename type::const_reverse_iterator const_reverse_iterator;
  typedef boost::intrusive::iterator_traits<iterator> it_traits;
  typedef boost::intrusive::iterator_traits<const_iterator> cit_traits;
  typedef boost::intrusive::iterator_traits<reverse_iterator> rit_traits;
  typedef boost::intrusive::iterator_traits<
    const_reverse_iterator> crit_traits;

  /* iterator_category */
  BOOST_STATIC_ASSERT(
    (boost::container::container_detail::is_same<
      std::random_access_iterator_tag,
      typename it_traits::iterator_category>::value));
  BOOST_STATIC_ASSERT(
    (boost::container::container_detail::is_same<
      std::random_access_iterator_tag,
      typename cit_traits::iterator_category>::value));
  BOOST_STATIC_ASSERT(
    (boost::container::container_detail::is_same<
      std::random_access_iterator_tag,
      typename rit_traits::iterator_category>::value));
  BOOST_STATIC_ASSERT(
    (boost::container::container_detail::is_same<
      std::random_access_iterator_tag,
      typename crit_traits::iterator_category>::value));

  test_iterator_random_and_compatible(c);
}

