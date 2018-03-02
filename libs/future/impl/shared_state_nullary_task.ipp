#ifndef SHARED_STATE_NULLARY_TASK_IPP
#define SHARED_STATE_NULLARY_TASK_IPP
#include <include/futures.hpp>

namespace boost {
#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
namespace detail {

template <typename R, typename F>
struct shared_state_nullary_task {
  typedef boost::shared_ptr<
    boost::detail::shared_state_base> storage_type;
  storage_type storage_;
  F f_;

  shared_state_nullary_task(storage_type storage, BOOST_THREAD_FWD_REF(F) f) :
    storage_(storage), f_(f) {}

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
  BOOST_THREAD_COPYABLE_AND_MOVABLE(shared_state_nullary_task)

  shared_state_nullary_task(shared_state_nullary_task const& s) :
    storage_(s.storage_), f_(s.f_) {}

  shared_state_nullary_task& operator=(
    BOOST_THREAD_COPY_ASSIGN_REF(
      shared_state_nullary_task) s) {
    if (this != &s) {
      storage_ = s.storage_;
      f_ = s.f_;
    }
    return *this;
  }

  shared_state_nullary_task(
    BOOST_THREAD_RV_REF(shared_state_nullary_task) s) :
    storage_(s.storage_), f_(boost::move(s.f_)) {
    s.storage_.reset();
  }

  shared_state_nullary_task& operator=(
    BOOST_THREAD_RV_REF(
      shared_state_nullary_task) s) {
    if (this != s) {
      storage_ = s.storage_;
      f_ = boost::move(f_);
      s.storage_.reset();
    }
    return *this;
  }
#endif // BOOST_NO_CXX11_RVALUE_REFERENCES

  void operator()() {
    boost::shared_ptr<boost::detail::shared_state<R> > storage =
      static_pointer_cast<boost::detail::shared_state<R> >(storage_);

    try {
      storage->mark_finished_with_result(f_());
    } catch (...) {
      storage->mark_exceptional_finish();
    }
  }
};

template <typename F>
struct shared_state_nullary_task<void, F> {
  typedef boost::shared_ptr<
    boost::detail::shared_state_base> storage_type;
  storage_type storage_;
  F f_;

  shared_state_nullary_task(storage_type storage, BOOST_THREAD_FWD_REF(F) f) :
    storage_(storage), f_(f) {}

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
  BOOST_THREAD_COPYABLE_AND_MOVABLE(shared_state_nullary_task)

  shared_state_nullary_task(shared_state_nullary_task const& s) :
    storage_(s.storage_), f_(s.f_) {}

  shared_state_nullary_task& operator=(
    BOOST_THREAD_COPY_ASSIGN_REF(
      shared_state_nullary_task) s) {
    if (this != s) {
      storage_ = s.storage_;
      f_ = s.f_;
    }
    return *this;
  }

  shared_state_nullary_task(
    BOOST_THREAD_RV_REF(shared_state_nullary_task) s) :
    storage_(s.storage_), f_(boost::move(s.f_)) {
    s.storage_.reset();
  }

  shared_state_nullary_task& operator=(
    BOOST_THREAD_RV_REF(
      shared_state_nullary_task) s) {
    if (this != s) {
      storage_ = s.storage_;
      f_ = boost::move(f_);
      s.storage_.reset();
    }
    return *this;
  }
#endif // BOOST_NO_CXX11_RVALUE_REFERENCES

  void operator()() {
    boost::shared_ptr<boost::detail::shared_state<void> > storage =
      static_pointer_cast<boost::detail::shared_state<void> >(storage_);

    try {
      f_();
      storage->mark_finished_with_result();
    } catch (...) {
      storage->mark_exceptional_finish();
    }
  }
};
} // detail

BOOST_THREAD_DCL_MOVABLE_BEG2(R, F)
boost::detail::shared_state_nullary_task<R, F>
BOOST_THREAD_DCL_MOVABLE_END

#endif // BOOST_THREAD_PROVIDES_EXECUTORS
} // boost

#endif // SHARED_STATE_NULL_TASK_IPP
