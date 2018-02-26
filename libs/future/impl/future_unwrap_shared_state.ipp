#ifndef FUTURE_UNWRAP_SHARED_STATE_IPP
#define FUTURE_UNWRAP_SHARED_STATE_IPP

#include <include/futures.hpp>

namespace boost {
#ifdef BOOST_THREAD_PROVIDES_FUTURE_UNWRAP
namespace detail {

template <typename F, typename R>
struct future_unwrap_shared_state :
  boost::detail::shared_state<R> {
  F wrapped_;
  typename F::value_type unwrapped_;

  explicit future_unwrap_shared_state(BOOST_THREAD_RV_REF(F) wrapped) :
    wrapped_(boost::move(wrapped)) {}

  void launch_continuation() {
    boost::unique_lock<boost::mutex> lock(this->mutex_);

    if (!unwrapped_.valid()) {
      if (unwrapped_.has_exception()) {
        this->mark_exceptional_finish_internal(
          wrapped_.get_exception_ptr(),
          lock);
      } else {
        unwrapped_ = wrapped_.get();

        if (unwrapped_.valid()) {
          lock.unlock();
          boost::unique_lock<boost::mutex> lock_(unwrapped_.future_->mutex_);
          unwrapped_.future_->set_continuation_ptr(
            this->shared_from_this(),
            lock_);
        } else {
          this->mark_exceptional_finish_internal(
            boost::copy_exception(
              boost::future_uninitialized()),
            lock);
        }
      }
    } else {
      if (unwrapped_.has_exception()) {
        this->mark_exceptional_finish_internal(
          unwrapped_.get_exception_ptr(),
          lock);
      } else {
        this->mark_finished_with_result_internal(lock);
      }
    }
  }
};

template <typename F>
struct future_unwrap_shared_state<F, void> :
  boost::detail::shared_state<void> {
  F wrapped_;
  typename F::value_type unwrapped_;

  explicit future_unwrap_shared_state(BOOST_THREAD_RV_REF(F) wrapped) :
    wrapped_(boost::move(wrapped)) {}

  void launch_continuation() {
    boost::unique_lock<boost::mutex> lock(this->mutex_);

    if (!unwrapped_.valid()) {
      if (wrapped_.has_exception()) {
        this->mark_exceptional_finish_internal(
          wrapped_.get_exception_ptr(),
          lock);
      } else {
        unwrapped_ = wrapped_.get();

        if (unwrapped_.valid()) {
          lock.unlock();
          boost::unique_lock<boost::mutex> lock_(unwrapped_.future_->mutex_);
          unwrapped_.future_->set_continuation_ptr(
            this->shared_from_this(),
            lock_);
        } else {
          this->mark_exceptional_finish_internal(
            boost::copy_exception(
              boost::future_uninitialized()),
            lock);
        }
      }
    } else {
      if (unwrapped_.has_exception()) {
        this->mark_exceptional_finish_internal(
          unwrapped_.get_exception_ptr(),
          lock);
      } else {
        this->mark_finished_with_result_internal(lock);
      }
    }
  }
};

template <typename F, typename R>
BOOST_THREAD_FUTURE<R> make_future_unwrap_shared_state(
  boost::unique_lock<boost::mutex>& lock,
  BOOST_THREAD_RV_REF(F) f) {
  boost::shared_ptr<
    boost::detail::future_unwrap_shared_state<F, R> > h(
      new boost::detail::future_unwrap_shared_state<F, R>(
        boost::move(f)));

  h->wrapped_.future_->set_continuation_ptr(h, lock);

  return BOOST_THREAD_FUTURE<R>(h);
}
} // detail

template <typename R>
inline BOOST_THREAD_FUTURE<R>::BOOST_THREAD_FUTURE(
  BOOST_THREAD_RV_REF(
    BOOST_THREAD_FUTURE<BOOST_THREAD_FUTURE<R> >) that) :
  base_type(that.unwrap()) {}

template <typename R>
BOOST_THREAD_FUTURE<R> BOOST_THREAD_FUTURE<BOOST_THREAD_FUTURE<R> >::unwrap() {
  BOOST_THREAD_PROVIDES_CONDITION(
    this->future_.get() != 0,
    boost::future_uninitialized());

  typedef BOOST_THREAD_FUTURE<R> R2;

  boost::shared_ptr<
    boost::detail::shared_state_base> shared_state(this->future_);
  boost::unique_lock<boost::mutex> lock(shared_state->mutex_);

  return boost::detail::make_future_unwrap_shared_state<
    BOOST_THREAD_FUTURE<R2>, R>(
      lock,
      boost::move(*this));
}
#endif // BOOST_THREAD_PROVIDES_FUTURE_UNWRAP

} // boost
#endif // FUTURE_UNWRAP_SHARED_STATE_IPP
