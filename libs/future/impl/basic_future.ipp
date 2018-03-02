#ifndef BASIC_FUTURE_IPP
#define BASIC_FUTURE_IPP
#include <include/futures.hpp>

namespace boost {
namespace detail {

template <typename R>
class basic_future : public base_future {
public:
  typedef typename boost::detail::shared_state<R> shared_state;
  typedef boost::shared_ptr<shared_state> future_ptr;
  typedef typename shared_state::move_dest_type move_dest_type;
  typedef boost::future_state::state state;
  typedef boost::detail::shared_state_base shared_state_base;
  typedef shared_state_base::notify_when_ready_handle notify_when_ready_handle;

  static future_ptr make_exceptional_future_ptr(
    boost::exceptional_ptr const& e) {
    return future_ptr(
      new boost::detail::shared_state<R>(e));
  }

  future_ptr future_;

  // Constructor
  basic_future(future_ptr future) : future_(future) {}
  basic_future(boost::exceptional_ptr const& e) :
    future_(make_exceptional_future_ptr(e)) {}
  BOOST_THREAD_MOVABLE_ONLY(basic_future) basic_future() : future_() {}

  // Copy constructor and assignment
  basic_future(BOOST_THREAD_RV_REF(basic_future) that) BOOST_NOEXCEPT {
    future_ = BOOST_THREAD_RV(that).future_;
    BOOST_THREAD_RV(that).future_.reset();
  }

  basic_future& operator=(
    BOOST_THREAD_RV_REF(basic_future) that) BOOST_NOEXCEPT  {
    future_ = BOOST_THREAD_RV(that).future_;
    BOOST_THREAD_RV(that).future_.reset();
    return *this;
  }

  void swap(basic_future& that) BOOST_NOEXCEPT {
    future_.swap(that.future_);
  }

  state get_state(
    boost::unique_lock<boost::mutex>& lock) const {
    if (!future_) {
      return boost::future_uninitialized();
    }
    return future_->get_state(lock);
  }

  state get_state() const {
    if (!future_) {
      return boost::future_state::uninitialized;
    }
    return future_->get_state();
  }

  bool is_ready(
    boost::unique_lock<boost::mutex>& lock) const {
    return get_state(lock) == boost::future_state::ready;
  }

  bool is_ready() const {
    return get_state() == boost::future_state::ready;
  }

  bool has_exception() const {
    return future_ && future_->has_exception();
  }

  bool has_value() const {
    return future_ && future_->has_value();
  }

  boost::launch launch_policy(
    boost::unique_lock<boost::mutex>& lock) const {
    if (future_) {
      return future_->launch_policy(lock);
    } else {
      return boost::launch(boost::launch::none);
    }
  }

  boost::launch launch_policy() const {
    if (future_) {
      boost::unique_lock<boost::mutex> lock(this->future_->mutex_);
      return future_->launch_policy(lock);
    } else {
      return boost::launch(boost::launch::none);
    }
  }

  boost::exception_ptr get_exception_ptr() {
    if (future_) {
      return future_->get_exception_ptr();
    } else {
      return boost::exception_ptr();
    }
  }

  bool valid() const {
    return future_ && future_->valid();
  }

  void wait() const {
    if (!future_) {
      boost::throw_exception(boost::future_uninitialized());
    }
    future_->wait(false);
  }

  boost::mutex& get_mutex() {
    if (!future_) {
      boost::throw_exception(boost::future_uninitialized());
    }
    return future_->mutex_;
  }

  notify_when_ready_handle notify_when_ready(
    boost::condition_variable_any& cv) {
    if (!future_) {
      boost::throw_exception(boost::future_uninitialized());
    }
    return future_->notify_when_ready(cv);
  }

  void unnotify_when_ready(
    notify_when_ready_handle h) {
    if (!future_) {
      boost::throw_exception(boost::future_uninitialized());
    }
    return future_->unnotify_when_ready(h);
  }

#ifdef BOOST_THREAD_USES_DATE
  template <typename Duration>
  bool timed_wait(
    Duration const& real_time) const {
    return timed_wait_until(boost::get_system_time() + real_time);
  }

  bool timed_wait_until(
    boost::system_time const& abs_time) const {
    if (!future_) {
      boost::throw_exception(boost::future_uninitialized());
    }
    return future_->timed_wait_until(abs_time);
  }
#endif // BOOST_THREAD_USES_DATE

#ifdef BOOST_THREAD_USES_CHRONO
  template <typename Rep, typename Period>
  boost::future_status wait_until(
    const boost::chrono::duration<Rep, Period>& real_time) const {
    return wait_until(boost::chrono::steady_clock::now() + real_time);
  }

  template <typename Clock, typename Duration>
  boost::future_status wait_unitl(
    const boost::chrono::time_point<Clock, Duration>& abs_time) const {
    if (!future_) {
      boost::throw_exception(boost::future_uninitialized());
    }
    return future_->wait_until(abs_time);
  }
#endif // BOOST_THREAD_USES_CHRONO
};
} // detail

BOOST_THREAD_DCL_MOVABLE_BEG(T)
boost::detail::basic_future<T>
BOOST_THREAD_DCL_MOVABLE_END

} // boost

#endif // BASIC_FUTURE_IPP
