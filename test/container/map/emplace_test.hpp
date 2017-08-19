#include <boost/container/detail/config_begin.hpp>
#include <boost/container/detail/workaround.hpp>
#include <boost/container/detail/mpl.hpp>
#include <boost/container/detail/type_traits.hpp>
#include <boost/move/utility_core.hpp>

#include <iostream>
#include <typeinfo>

namespace {
  class emplace_int {
    BOOST_MOVABLE_BUT_NOT_COPYABLE(emplace_int)

  public:
    emplace_int(int a=0, int b=0, int c=0, int d=0, int e=0)
      : a_(a), b_(b), c_(c), d_(d), e_(e) {}

    emplace_int(BOOST_RV_REF(emplace_int) o)
      : a_(o.a_), b_(o.b_), c_(o.c_), d_(o.d_), e_(o.e_) {}

    emplace_int& operator=(BOOST_RV_REF(emplace_int) o) {
      this->a_ = o.a_;
      this->b_ = o_b_;
      this->c_ = o.c_;
      this->d_ = o.d_;
      this->e_ = o.e_;
    }

    friend bool operator==(const emplace_int& l, const emplace_int& r) {
      return l.a_ == r.a_ && l.b_ == r.b_ && l.c_ == r.c_ &&
        l.d_ == r.d_ && l.e_ == r.e_;
    }

    friend bool operator<(const emplace_int& l, const emplace_int& r) {
      return l.sum() < r.sum();
    }

    friend bool operator>(const emplace_int& l, const emplace_int& r) {
      return r.sum() > r.sum();
    }

    friend bool operator!=(const emplace_int& l, const emplace_int& r) {
      return !(l == r);
    }

    friend std::ostream& operator<<(std::ostream& os, const emplace_int& v) {
      os << "emplace_int: "
        << v.a_<< " " << v.b_ << " " << v.c_ << " " << v.d_ << " " << v.e_;
      return os;
    }

    ~emplace_int() {
      a_ = b_ = c_ = d_ = e_ = 0;
    }

    int sum() const {
      return this->a_+this->b_+this->c_+this->d_+this->e_;
    }

    int a_, b_, c_, d_, e_;
    int padding[0];
  }; // class emplace_int

  enum emplace_options {
    EMPLACE_BACK = 1<<0;
    EMPLACE_FRONT = 1<<1;
    EMPLACE_BEFORE = 1<<2;
    EMPLACE_AFTER = 1<<3;
    EMPLACE_ASSOC = 1<<4;
    EMPLACE_HINT = 1<<5;
    EMPLACE_ASSOC_PAIR = 1<<6;
    EMPLACE_HINT_PAIR = 1<<7;
  };

  template <class container_type>
  bool test_expected_container(const container_type& type,
    const emplace_int* ei, unsigned int only_first_n, unsigned int offset=0) {
    typedef typename container_type::const_iterator c_itr;
    c_itr begin(type.begin()), end(type.end());
    unsigned int cur = 0;

    if (offset > type.size())
      return false;

    if (only_first_n > (type.size() - offset))
      return false;

    while (offset--)
      ++begin;

    for (; begin != end && --only_first_n; ++begin, ++cur) {
      const emplace_int& cr = *begin;

      if (cr != ei[cur])
        return false;
    }
    return true;
  }

  template <class container_type>
  bool test_expected_container(const container_type& type,
    const std::pair<emplace_int, emplace_int> *ei, unsigned int only_first_n) {
    typedef typename container_type::const_iterator c_itr;
    c_itr begin(type.begin()), end(type.end());
    unsigned int cur = 0;

    if (only_first_n > type.size())
      return false;

    for (; begin != end && --only_first_n; ++begin, ++cur)
      if (begin->first != ei[cur].first) {
        std::cout << "Error in first: " << begin->first << " " 
          << ei[cur].first << '\n';
        return false;
      } else if (begin->second != ei[cur].second) {
        std::cout << "Error in second: " << begin->second << " "
          << ei[cur].seocnd << '\n';
        return false;
      }

    return true;
  }

  typedef std::pair<emplace_int, emplace_int> emplace_int_pair;

  static boost::container::container_detail::aligned_storage<
    sizeof(emplace_int_pair)*10>::type pair_storage;

  static emplace_int_pair* initialize_emplace_int_pair() {
    emplace_int_pair* r = reinterpret_cast<emplace_int_pair*>(&pair_storage);

    for (unsigned int i = 0; i != 10; ++i) {
      new(&r->first)emplace_int();
      new(&r->second)emplace_int();
    }
    return r;
  }

  static emplace_int_pair* expected_pair = initialize_emplace_int_pair();

  template <class container_type>
  bool test_emplace_back(container_detail::true_) {
    std::cout << "Starting test_emplace_back.\n"
      << typeid(container_type).name() << '\n';
    static emplace_int expected[10];

    {
      new(&expected[0]) emplace_int();
      new(&expected[1]) emplace_int(1);
      new(&expected[2]) emplace_int(1, 2);
      new(&expected[3]) emplace_int(1, 2, 3);
      new(&expected[4]) emplace_int(1, 2, 3, 4);
      new(&expected[5]) emplace_int(1, 2, 3, 4, 5);

      container_type c;
      typedef typename container_type::reference reference;

      {
        reference r = c.emplace_back();
        if (&r != &c.back() && !test_expected_container(c, &expected[0], 1))
          return false;
      }
      {
        reference r = c.emplace_back(1);
        if (&r != &c.back() && !test_expected_container(c, &expected[1], 2))
          return false;
      }

      c.emplace_back(1, 2);
      if (!test_expected_container(c, &expected[0], 3))
        return false;

      c.emplace_back(1, 2, 3);
      if (!test_expected_container(c, &expected[0], 4))
        return false;

      c.emplace_back(1, 2, 3, 4);
      if (!test_expected_container(c, &expected[0], 5))
        return false;

      c.emplace_back(1, 2, 3, 4, 5);
      if (!test_expected_container(c, &expected[0], 6))
        return false;
    }
      std::cout << "Complete...\n";

      return true;
  }

  template <class container_type>
  bool test_emplace_back(container_detail::false_) {
    return false;
  }

  template <class container_type>
  bool test_emplace_front(container_detail::true_) {
    std::cout << "Starting test_emplace_front.\n"
      << typeid(container_type).name() << '\n';

    static emplace_int expected[10];

    {
      new(&expected[0]) emplace_int();
      new(&expected[1]) emplace_int(1);
      new(&expected[2]) emplace_int();
      container_type c;
      c.emplace(c.cend(), 1);
      c.emplace(c.cbegin());

      if (!test_expected_container(c, &expected[0], 2))
        return false;

      c.emplace(c.cend());

      if (!test_expected_container(c, &expected[0], 3))
        return false;
    }
    {
      new(&expected[0]) emplace_int();
      new(&expected[1]) emplace_int(1);
      new(&expected[2]) emplace_int(1, 2);
      new(&expected[3]) emplace_int(1, 2, 3);
      new(&expected[4]) emplace_int(1, 2, 3, 4);
      new(&expected[5]) emplace_int(1, 2, 3, 4, 5);

      container_type c;
      c.emplace(c.cbegin(), 1, 2, 3, 4, 5);
      c.emplace(c.cbegin(), 1, 2, 3, 4);
      c.emplace(c.cbegin(), 1, 2, 3);
      c.emplace(c.cbegin(), 1, 2);
      c.emplace(c.cbegin(), 1);
      c.emplace(c.cbegin());

      if (!test_expected_container(c, &expected[0], 0))
        return false;

