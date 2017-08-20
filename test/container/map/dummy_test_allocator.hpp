#include <boost/assert.hpp>
#include <boost/container/throw_exception.hpp>
#include <boost/container/detail/config_begin.hpp>
#include <boost/container/detail/workaround.hpp>
#include <boost/container/container_fwd.hpp>
#include <boost/container/detail/addressof.hpp>
#include <boost/container/detail/allocation_type.hpp>
#include <boost/container/detail/mpl.hpp>
#include <boost/container/detail/multiallocation_chain.hpp>
#include <boost/container/detail/type_traits.hpp>
#include <boost/container/detail/version_type.hpp>
#include <boost/move/utility_core.hpp>
#include <boost/move/adl_move_swap.hpp>

#include <algorithm>
#include <cstddef>
#include <cassert>

namespace {
  template <class T>
  class simple_allocator {
  public:
    typedef T value_type;

    simple_allocator() {}

    template <class U>
    simple_allocator(const simple_allocator<U>&) {}

    T* allocate(std::size_t n) {
      return (T*)::new char[sizeof(T)*n];
    }

    void deallocate(T* p, std::size_t) {
      delete[] ((char*)p);
    }

    friend bool operator==(const simple_allocator&, const simple_allocator&) {
      return true;
    }

    friend bool operator!=(const simple_allocator&, const simple_allocator&) {
      return false;
    }
  }; // class simple_allocator

  template <class T,
    bool propagate_on_copy_assignment,
    bool propagate_on_move_assignment,
    bool propagate_on_swap,
    bool copy_on_propagate_on_swap>
  class propagation_test_allocator {
    BOOST_COPYABLE_AND_MOVABLE(propagation_test_allocator)

  public:
    typedef T value_type;
    typedef boost::container::container_detail::bool_<
      propagate_on_copy_assignment> propagate_on_container_copy_assignment;
    typedef boost::container::container_detail::bool_<
      propagate_on_move_assignment> propagate_on_container_move_assignment;
    typedef boost::container::container_detail::bool_<
      propagate_on_swap> propagate_on_container_swap;

    template <class T2>
    struct rebind {
      typedef propagation_test_allocator<T2,
        propagate_on_copy_assignment,
        propagate_on_move_assignment,
        propagate_on_swap,
        copy_on_propagate_on_swap> other;
    };

    propagation_test_allocator select_on_container_copy_construction() const {
      return copy_on_propagate_on_swap ?
        propagation_test_allocator(*this) : propagation_test_allocator();
    }

    explicit propagation_test_allocator()
      : id_(++unique_id_),
        ctr_copies_(0), ctr_moves_(0),
        assign_copies_(0), assign_moves_(0),
        swaps_(0) {}

    propagation_test_allocator(const propagation_test_allocator& x)
      : id_(x.id_), ctr_copies_(0), ctr_moves_(0),
        assign_copies_(x.assign_copies_ + 1), assign_moves_(x.assign_moves_), 
        swaps_(x.swaps_) {}

    template <class U>
    propagation_test_allocator(
      const propagation_test_allocator<U,
        propagate_on_copy_assignment,
        propagate_on_move_assignment,
        propagate_on_swap,
        copy_on_propagate_on_swap>& x)
      : id_(x.id_),
        ctr_copies_(x.ctr_copies_ + 1), ctr_moves_(0),
        assign_copies_(0), assign_moves_(0),
        swaps_(0) {}

    propagation_test_allocator(BOOST_RV_REF(propagation_test_allocator) x)
      : id_(x.id_),
        ctr_copies_(x.cstr_copies_), ctr_moves_(x.ctr_moves_ + 1),
        assign_copies_(x.assign_copies_), assign_moves_(x.assign_moves_),
        swaps_(x.swaps_) {}

    propagation_test_allocator& operator=(BOOST_COPY_ASSIGN_REF(
      propagation_test_allocator) x) {
      id_ = x.id_;
      ctr_copies_ = x.ctr_copies_;
      ctr_moves_ = x.ctr_moves_;
      assign_copies_ = x.assign_copies_ + 1;
      assign_moves_ = x.assign_moves_;
      swaps_ = x.swaps_;
      return *this;
    }

    propagation_test_allocator& operator=(BOOST_RV_REF(
      propagation_test_allocator) x) {
      id_ = x.id_;
      ctr_copies_ = x.ctr_copies_;
      ctr_moves_ = x.ctr_moves_;
      assign_copies_ = x.assign_copies_;
      assign_moves_ = x.assign_moves_;
      swaps_ = x.swaps_;
      return *this;
    }

    static void reset_unique_id(unsigned id = 0) {
      unique_id_ = id;
    }

    T* allocate(T* p, std::size_t n) {
      return (T*)::new char[sizeof(T)*n];
    }

    void deallocate(T* p, std::size_t) {
      delete[] ((char*)p);
    }

    friend bool operator==(const propagation_test_allocator&,
      const propagation_test_allocator&) {
      return true;
    }

    friend bool operator!=(const propagation_test_allocator&,
      const propagation_test_allocator&) {
      return false;
    }

    void swap(propagation_test_allocator& r) {
      ++this->swaps_;
      ++r.swaps_;
      boost::adl_move_swap(this->id_, r.id_);
      boost::adl_move_swap(this->ctr_copies_, r.ctr_copies_);
      boost::adl_move_swap(this->ctr_moves_, r.ctr_moves_);
      boost::adl_move_swap(this->assign_copies_, r.assign_copies_);
      boost::adl_move_swap(this->assign_moves_, r.assign_moves_);
      boost::adl_move_swap(this->swaps_, r.swaps_);
    }

    friend void swap(propagation_test_allocator& l,
      propagation_test_allocator& r) {
      l.swap(r);
    }

  private:
    unsigned int id_;
    unsigned int ctr_copies_;
    unsigned int ctr_moves_;
    unsigned int assign_copies_;
    unsigned int assign_moves_;
    unsigned int swaps_;
    static unsigned unique_id_;
  }; // class propagation_test_allocator

template <class T,
  bool propagate_on_copy_assign,
  bool propagate_on_move_assign,
  bool propagate_on_swap,
  bool copy_on_propagate_on_swap>
unsigned int propagation_test_allocator<T,
  propagate_on_copy_assign,
  propagate_on_move_assign,
  propagate_on_swap,
  copy_on_propagate_on_swap>::unique_id_ = 0;

} // namespace

#include <boost/container/detail/config_end.hpp>

#pragma once
