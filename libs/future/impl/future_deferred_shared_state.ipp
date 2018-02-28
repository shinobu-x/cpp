#ifndef FUTURE_DEFERRED_SHARED_STATE_IPP
#define FUTURE_DEFERRED_SHARED_STATE_IPP
#include <include/futures.hpp>

namespace boost {
namespace detail {

template <typename S, typename F>
struct future_deferred_shared_state :
  boost::detail::shared_state<S> {
  typedef boost::detail::shared_state<S> base_type;
  F f_;

  // Constructor
  explicit future_deferred_shared_state(BOOST_THREAD_FWD_REF(F) f) :
    f_(boost::move(f)) {
    this->set_deferred();
  }

  virtual void execute(
    boost::unique_lock<boost::mutex>& lock) {
    try {
      F f(boost::move(f_));
      boost::detail::relocker relock(lock);
      S r = f();
      relock.lock();
      this->mark_finished_with_result_internal(boost::move(r), lock);
    } catch (...) {
      this->mark_exceptional_finish_internal(
        boost::current_exception(), lock);
    }
  }
};

template <typename S, typename F>
struct future_deferred_shared_state<S&, F> :
  boost::detail::shared_state<S&> {
  typedef boost::detail::shared_state<S&> base_type;
  F f_;

  // Constructor
  explicit future_deferred_shared_state(BOOST_THREAD_FWD_REF(F) f) :
    f_(boost::move(f)) {
    this->set_deferred();
  }

  virtual void execute(
    boost::unique_lock<boost::mutex>& lock) {
    try {
      this->mark_finished_with_result_internal(f_(), lock);
    } catch (...) {
      this->mark_exceptional_finish_internal(
        boost::current_exception(), lock);
    }
  }
};

template <typename F>
struct future_deferred_shared_state<void, F> :
  boost::detail::shared_state<void> {
  typedef boost::detail::shared_state<void> base_type;
  F f_;

  // Constructor
  explicit future_deferred_shared_state(BOOST_THREAD_FWD_REF(F) f) :
    f_(boost::move(f)) {
    this->set_deferred();
  }

  virtual void execute(
    boost::unique_lock<boost::mutex>& lock) {
    try {
      F f = boost::move(f_);
      relocker relock(lock);
      f();
      relock.lock();
      this->mark_finished_with_result_internal(lock);
    } catch (...) {
      this->mark_exceptional_finish_internal(
        boost::current_exception(), lock);
    }
  }
};

} // detail
} // boost

#endif // FUTURE_DEFERRED_SHARED_STATE_IPP
