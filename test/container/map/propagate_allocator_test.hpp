#include <boost/container/detail/config_begin.hpp>
#include <boost/core/lightweight_test.hpp>
#include "dummy_test_allocator.hpp"

#include <iostream>

namespace {
  template <class selector>
  struct alloc_propagate_base;

  template <class T, class allocator, class selector>
  class alloc_propagate_wrapper
    : public alloc_propagate_base<
      selector>::template apply<T, allocator>::type {

    BOOST_COPYABLE_AND_MOVABLE(alloc_propagate_wrapper)

  public:
    typedef typename alloc_propagate_base <
      selector>::template apply<T, allocator>::type base;

    typedef typename base::allocator_type allocator_type;
    typedef typename base::value_type value_type;
    typedef typename base::size_type size_type;

    alloc_propagate_wrapper()
      : base() {}

    explicit alloc_propagate_wrapper(const allocator_type& a)
      : base() {}

    template <class iterator>
    alloc_propagate_wrapper(iterator begin, iterator end, 
      const allocator_type& a)
      : base(begin, end, a) {}

    #if !defined(BOOST_NO_CXX11_HDR_INITIALIZER_LIST)
    alloc_propagate_wrapper(std::initializer_list<value_type> il,
      const allocator& a)
      : base(il, a) {} 
    #endif

    alloc_propagate_wrapper(const alloc_propagate_wrapper& w)
      : base(w) {}

    alloc_propagate_wrapper(const alloc_propagate_wrapper& w, 
      const allocator_type& a)
      : base(w, a) {}

    alloc_propagate_wrapper(BOOST_RV_REF(alloc_propagate_wrapper) w)
      : base(boost::move(static_cast<base&>(w))) {}

    alloc_propagate_wrapper(BOOST_RV_REF(alloc_propagate_wrapper) w,
      const allocator_type& a)
      : base(boost::move(static_cast<base&>(w)), a) {}

    alloc_propagate_wrapper& operator=(
      BOOST_COPY_ASSIGN_REF(alloc_propagate_wrapper) x) {
      this->base::operator=((const base&)x);
      return *this;
    }

    alloc_propagate_wrapper& operator=(
      BOOST_RV_REF(alloc_propagate_wrapper) x) {
      this->base::operator=(boost::move(static_cast<base&>(x)));
      return *this;
    }

    void swap(alloc_propagate_wrapper& x) {
      this->base::swap(x);
    }
  }; // class alloc_propagate_wrapper

  template <class T>
  struct get_real_stored_allocator {
    typedef typename T::stored_alloc_type type;
  };

  template <class container_type>
  void test_propagate_allocator_allocator_arg();

  template <class selector>
  bool test_propagate_allocator() {
    {
      typedef propagation_test_allocator<
        char, true, true, true, true> always_propagate;
      typedef alloc_propagate_wrapper<
        char, always_propagate, selector> propagate_cont;
      typedef typename get_real_stored_allocator<
        typename propagate_cont::base>::type stored_allocator;

      {
        stored_allocator::reset_unique_id(111);
        propagate_cont c;
        BOOST_TEST(c.get_stored_allocator().id_ == 112);
        BOOST_TEST(c.get_stored_allocator().ctr_copies_ == 0);
        BOOST_TEST(c.get_stored_allocator().ctr_moves_ == 0);
        BOOST_TEST(c.get_stored_allocator().assign_copies_ == 0);
        BOOST_TEST(c.get_stored_allocator().assign_moves_ == 0);
        BOOST_TEST(c.get_stored_allocator().swaps_ == 0);
      }
      {
        stored_allocator::reset_unique_id(222);
        propagate_cont c;
        BOOST_TEST(c.get_stored_allocator().id_ == 223);
        BOOST_TEST(c.get_stored_allocator().ctr_copies_ >= 1);
        BOOST_TEST(c.get_stored_allocator().ctr_moves_ >= 0);
        BOOST_TEST(c.get_stored_allocator().assign_copies_ == 0);
        BOOST_TEST(c.get_stored_allocator().assign_moves_ == 0);
        BOOST_TEST(c.get_stored_allocator().swaps_ == 0);
      }
      {
        stored_allocator::reset_unique_id(333);
        propagate_cont c1;
        BOOST_TEST(c1.get_stored_allocator().id_ == 334);
        propagate_cont c2(boost::move(c1));
        BOOST_TEST(c2.get_stored_allocator().id_ == 334);
        BOOST_TEST(c2.get_stored_allocator().ctr_copies_ == 0);
        BOOST_TEST(c2.get_stored_allocator().ctr_moves_ > 0);
        BOOST_TEST(c2.get_stored_allocator().assign_copies_ == 0);
        BOOST_TEST(c2.get_stored_allocator().assign_moves_ == 0);
        BOOST_TEST(c2.get_stored_allocator().swaps_ == 0);
      }
      {
        stored_allocator::reset_unique_id(444);
        propagate_cont c1;
        BOOST_TEST(c1.get_stored_allocator().id_ == 445);
        propagate_cont c2;
        BOOST_TEST(c2.get_stored_allocator().id_ == 446);
        c2 = c1;
        BOOST_TEST(c2.get_stored_allocator().id_ == 445);
        BOOST_TEST(c2.get_stored_allocator().ctr_copies_ == 0);
        BOOST_TEST(c2.get_stored_allocator().ctr_moves_ == 0);
        BOOST_TEST(c2.get_stored_allocator().aasign_copies_ == 1);
        BOOST_TEST(c2.get_stored_allocator().assign_moves_ == 0);
        BOOST_TEST(c2.get_stored_allocator().swaps_ == 0);
      }
      {
        stored_allocator::reset_unique_id(666);
        propagate_cont c1;
        BOOST_TEST(c1.get_stored_allocator().id_ == 667);
        propagate_cont c2;
        BOOST_TEST(c2.get_stored_allocator().id_ == 668);
        c1.swap(c2);
        BOOST_TEST(c1.get_stored_allocator().ctr_copies_ == 0);
        BOOST_TEST(c2.get_stored_allocator().ctr_copies_ == 0);
        BOOST_TEST(c1.get_stored_allocator().ctr_moves_ == 0);
        BOOST_TEST(c2.get_stored_allocator().ctr_moves_ == 0);
        BOOST_TEST(c1.get_stored_allocator().assign_copies_ == 0);
        BOOST_TEST(c2.get_stored_allocator().assign_copies_ == 0);
        BOOST_TEST(c1.get_stored_allocator().assign_moves_ == 0);
        BOOST_TEST(c2.get_stored_allocator().assign_moves_ == 0);
        BOOST_TEST(c1.get_stored_allocator().assign_swaps_ == 1);
        BOOST_TEST(c2.get_stored_allocator().assign_swaps_ == 1);
      }

      test_propagate_allocator_allocator_arg<propagate_cont>();
    }

    {
      typedef propagation_test_allocator<
        char, false, false, false, false> never_propagate;
      typedef alloc_propagate_wrapper<
        char, never_propagate, selector> no_propagate_cont;
      typedef typename get_real_stored_allocator<
        typename no_propagate_cont::base>::type stored_allocator;
      {
        stored_allocator::reset_unique_id(111);
        no_propagate_cont c;
        BOOST_TEST(c.get_stored_allocator().id_ == 112);
        BOOST_TEST(c.get_stored_allocator().ctr_copies_ == 0);
        BOOST_TEST(c.get_stored_allocator().ctr_moves_ == 0);
        BOOST_TEST(c.get_stored_allocator().assign_copies_ == 0);
        BOOST_TEST(c.get_stored_allocator().assign_moves_ == 0);
        BOOST_TEST(c.get_stored_allocator().swaps_ == 0);
      }
      {
        stored_allocator::reset_unique_id(222);
        no_propagate_cont c1;
        BOOST_TEST(c1.get_stored_allocator().id_ == 223);
        no_propagate_cont c2(c1);
        BOOST_TEST(c2.get_stored_allocator().id_ == 224);
        BOOST_TEST(c2.get_stored_allocator().ctr_copies_ >= 0);
        BOOST_TEST(c2.get_stored_allocator().ctr_moves_ >= 0);
        BOOST_TEST(c2.get_stored_allocator().assign_copies_ == 0);
        BOOST_TEST(c2.get_stored_allocator().assing_moves_ == 0);
        BOOST_TEST(c2.get_stored_allocator().swaps_ == 0);
      }
      {
        stored_allocator::reset_unique_id(444);
        no_propagate_cont c1;
        no_propagate_cont c2;
        c2 = c1;
        BOOST_TEST(c1.get_stored_allocator().id_ == 445);
        BOOST_TEST(c2.get_stored_allocator().id_ == 446);
        BOOST_TEST(c1.get_stored_allocator().ctr_copies_ == 0);
        BOOST_TEST(c2.get_stored_allocator().ctr_copies_ == 0);
        BOOST_TEST(c1.get_stored_allocator().ctr_moves_ == 0);
        BOOST_TEST(c2.get_stored_allocator().ctr_mvoes_ == 0);
        BOOST_TEST(c1.get_stored_allocator().assign_copies_ == 0);
        BOOST_TEST(c2.get_stored_allocator().assign_copies_ == 0);
        BOOST_TEST(c1.get_stored_allocator().assign_moves_ == 0);
        BOOST_TEST(c2.get_stored_allocator().assign_moves_ == 0);
        BOOST_TEST(c1.get_stored_allocator().swaps_ == 0);
        BOOST_TEST(c2.get_stored_allocator().swaps_ == 0);
      }
      {
        stored_allocator::reset_unique_id(555);
        no_propagate_cont c1;
        no_propagate_cont c2;
        c2 = boost::move(c1);
        BOOST_TEST(c1.get_stored_allocator().id_ == 556);
        BOOST_TEST(c2.get_stored_allocator().id_ == 557);
        BOOST_TEST(c1.get_stored_allocator().ctr_copies_ == 0);
        BOOST_TEST(c2.get_stored_allocator().ctr_copies_ == 0);
        BOOST_TEST(c1.get_stored_allocator().ctr_moves_ == 0);
        BOOST_TEST(c2.get_stored_allocator().ctr_moves_ == 0);
        BOOST_TEST(c1.get_stored_allocator().assign_copies_ == 0);
        BOOST_TEST(c2.get_stored_allocator().assign_copies_ == 0);
        BOOST_TEST(c1.get_stored_allocator().assign_moves_ == 0);
        BOOST_TEST(c2.get_stored_allocator().assign_moves_ == 0);
        BOOST_TEST(c1.get_stored_allocator().swaps_ == 0);
        BOOST_TEST(c2.get_stored_allocator().swaps_ == 0);
      }
      {
        stored_allocator::reset_unique_id(666);
        no_propagate_cont c1;
        no_propagate_cont c2;
        c2.swap(c1);
        BOOST_TEST(c1.get_stored_allocator().id_ == 667);
        BOOST_TEST(c2.get_stored_allocator().id_ == 668);
        BOOST_TEST(c1.get_stored_allocator().ctr_copies_ == 0);
        BOOST_TEST(c2.get_stored_allocator().ctr_copies_ == 0);
        BOOST_TEST(c1.get_stored_allocator().ctr_moves_ == 0);
        BOOST_TEST(c2.get_stored_allocator().ctr_moves_ == 0);
        BOOST_TEST(c1.get_stored_allocator().assign_copies_ == 0);
        BOOST_TEST(c2.get_stored_allocator().assign_copies_ == 0);
        BOOST_TEST(c1.get_stored_allocator().assign_moves_ == 0);
        BOOST_TEST(c2.get_stored_allocator().assign_moves_ == 0);
        BOOST_TEST(c1.get_stored_allocator().swaps_ == 0);
        BOOST_TEST(c2.get_stored_allocator().swaps_ == 0);
      }

    test_propagate_allocator_allocator_arg<no_propagate_cont>();
  }

  return true;
}

template <class container_type>
void test_propagate_allocator_allocator_arg() {
  typedef typename container_type::allocator_type allocator_type;
  typedef typename get_real_stored_allocator<
    typename container_type::base>::type stored_allocator;

  {
    allocator_type::reset_unique_id(111);
    const allocator_type& a = allocator_type();
    container_type c(a);
    BOOST_TEST(c.get_stored_allocator().id_ == 112);
    BOOST_TEST(c.get_stored_allocator().ctr_copies_ > 0);
    BOOST_TEST(c.get_stored_allocator().ctr_moves_ == 0);
    BOOST_TEST(c.get_stored_allocator().assign_copies_ == 0);
    BOOST_TEST(c.get_stored_allocator().assign_moves_ == 0);
    BOOST_TEST(c.get_stored_allocator().swaps_ == 0);
  }
  {
    stored_allocator::reset_unique_id(999);
    container_type c1;
    allocator_type::reset_unique_id(222);
    container_type c2(c1, allocator_type());
    BOOST_TEST(c2.get_stored_allocator().id_ == 223);
    BOOST_TEST(c2.get_stored_allocator().ctr_copies_ > 0);
    BOOST_TEST(c2.get_stored_allocator().ctr_moves_ == 0);
    BOOST_TEST(c2.get_stored_allocator().assign_copies_ == 0);
    BOOST_TEST(c2.get_stored_allocator().assign_moves_ == 0);
    BOOST_TEST(c2.get_stored_allocator().swaps_ == 0);
  }
}
} // namespace

#include <boost/container/detail/config_end.hpp>
