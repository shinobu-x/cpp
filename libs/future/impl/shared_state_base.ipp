#ifndef SHARED_STATE_BASE_IPP
#define SHARED_STATE_BASE_IPP

#include "../include/futures.hpp"

namespace boost {
namespace detail {

struct shared_state_base :
  boost::enable_shared_from_this<shared_state_base> {

  typedef std::list<boost::condition_variable_any*> waiter_list;
  typedef waiter_list::iterator notify_when_ready_handle;
  typedef boost::shared_ptr<shared_state_base> continuation_ptr_type;
  typedef std::vector<continuation_ptr_type> continuations_type;

  bool done_;
  bool is_valid_;
  bool is_deferred_;
  bool is_constructed_;
  boost::launch policy_;
  executor_ptr_type ex_;
  mutable boost::mutex mutex_;
  waiter_list external_waiters_;
  boost::exception_ptr exception_;
  boost::function<void()> callback_;
  continuations_type continuations_;
  boost::condition_variable waiters_;

  virtual void launch_continuation() {}

  shared_state_base() :
    done_(false),
    is_valid_(true),
    is_deferred_(false),
    is_constructed_(false),
    policy_(boost::launch::none),
    continuations_(),
    ex_() {}

  shared_state_base(boost::exceptional_ptr const& e) :
    exception_(e.ptr_),
    done_(true),
    is_valid_(true),
    is_deferred_(false),
    is_constructed_(false),
    policy_(boost::launch::none),
    ex_() {}

  virtual ~shared_state_base() {}

  executor_ptr_type get_executor() {

    return ex_;

  }

  void set_executor_policy(executor_ptr_type ex) {

    set_executor();
    ex_ = ex;

  }

  void set_executor_policy(executor_ptr_type ex,
    boost::lock_guard<boost::mutex>&) {

    set_executor();
    ex_ = ex;

  }

  void set_executor_policy(executor_ptr_type ex,
    boost::unique_lock<boost::mutex>&) {

    set_executor();
    ex_ = ex;

  }

  bool valid(boost::unique_lock<boost::mutex>&) {

    return is_valid_;

  }

  bool valid() {

    boost::unique_lock<boost::mutex> lock(this->mutex_);
    valid(lock);

  }

  void invalidate(boost::unique_lock<boost::mutex>&) {

    is_valid_ = false;

  }

  void invalidate() {

    boost::unique_lock<boost::mutex> lock(this->mutex_);
    invalidate(lock);

  }

  void validate(boost::unique_lock<boost::mutex>&) {

    is_valid_ = true;

  }

  void validate() {

    boost::unique_lock<boost::mutex> lock(this->mutex_);
    validate(lock);

  }

  void set_async() {

    is_deferred_ = false;
    policy_ = boost::launch::async;

  }

  void set_deferred() {

    is_deferred_ = true;
    policy_ = boost::launch::deferred;

  }

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
  void set_executor() {

    is_deferred_ = false;
    policy_ = boost::launch::executor;

  }
#else
  void set_executor() {}
#endif // BOOST_THREAD_PROVIDES_EXECUTORS

  notify_when_ready_handle notify_when_ready(
    boost::condition_variable_any& cv) {

    boost::unique_lock<boost::mutex> lock(this->mutex_);
    do_callback(lock);

    return external_waiters_.insert(external_waiters_.end(), &cv);

  }

  void unnotify_when_ready(notify_when_ready_handle waiter) {

    boost::lock_guard<boost::mutex> lock(this->mutex_);
    external_waiters_.erase(waiter);

  }

#ifdef BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
  void do_continuation(boost::unique_lock<boost::mutex>& lock) {

    if (!continuations_.empty()) {

      continuations_type continuations = continuations_;
      continuations_.clear();
      relocker relock(lock);
      continuations_type::iterator it = continuations.begin();

      for (; it != continuations.end(); ++it) {
        (*it)->launch_continuation();
      }
    }

  }
#else
  void do_continuation(boost::unique_lock<boost::mutex>&) {}
#endif // BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION

#ifdef BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
  virtual void set_continuation_ptr(continuation_ptr_type continuation,
    boost::unique_lock<boost::mutex>& lock) {

    continuations_.push_back(continuation);
    if (done_) {
      do_continuation(lock);
    }

  }
#endif // BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION

  void mark_finished_internal(boost::unique_lock<boost::mutex>& lock) {

    done_ = true;
    waiters_.notify_all();
    waiter_list::const_iterator it = external_waiters_.begin();

    for (; it != external_waiters_.end(); ++it) {
      (*it)->notify_all();
    }
    do_continuation(lock);

  }

  void notify_deferred() {

    boost::unique_lock<boost::mutex> lock(this->mutex_);

  }

  void do_callback(boost::unique_lock<boost::mutex>& lock) {

    if (callback_ && !done_) {
      boost::function<void()> callback = callback_;
      relocker relock(lock);
      callback();
    }

  }

  virtual bool run_if_is_deferred() {

    boost::unique_lock<boost::mutex> lock(this->mutex_);
    if (is_deferred_) {
      is_deferred_ = false;
      execute(lock);
      return true;
    } else {
      return false;
    }

  }

  virtual bool run_if_is_deferred_or_ready() {

    boost::unique_lock<boost::mutex> lock(this->mutex_);
    if (is_deferred_) {
      is_deferred_ = false;
      execute(lock);
      return true;
    } else {
      return done_;
    }

  }

  void wait_internal(boost::unique_lock<boost::mutex>& lock,
    bool rethrow = true) {

    do_callback(lock);
    if (is_deferred_) {
      is_deferred_ = false;
      execute(lock);
    }

    while (!done_) {
      waiters_.wait(lock);
    }

    if (rethrow && exception_) {
      boost::rethrow_exception(exception_);
    }

  }

  virtual void wait(boost::unique_lock<boost::mutex>& lock,
    bool rethrow = true) {

    wait_internal(lock, rethrow);

  }

  void wait(bool rethrow = true) {

    boost::unique_lock<boost::mutex> lock(this->mutex_);
    wait(lock, rethrow);

  }

#ifdef BOOST_THREAD_USES_DATETIME
  bool timed_wait_until(boost::system_time const& target_time) {

    boost::unique_lock<boost::mutex> lock(this->mutex_);
    if (is_deferred_) {
      return false;
    }
    do_callback(lock);

    while (!done_) {
      bool const success = waiters_.timed_wait(lock, target_time);
      if (!success && !done_) {
        return false;
      }
    }
    return true;
  }
#endif // BOOST_THREAD_USES_DATETIME

#ifdef BOOST_THREAD_USES_CHRONO
  template <typename Clock, typename Duration>
  boost::future_status wait_until(
    const boost::chrono::time_point<Clock, Duration>& abs_time) {

    boost::unique_lock<boost::mutex> lock(this->mutex_);
    if (is_deferred_) {
      return boost::future_status::deferred;
    }
    do_callback(lock);

    while (!done_) {
      boost::cv_status const status = waiters_.wait_until(lock, abs_time);
      if (status == boost::cv_status::timeout && !done_) {
        return boost::future_status::timeout;
      }
    }
    return boost::future_status::ready;

  }
#endif // BOOST_THREAD_USES_CHRONO

  void mark_exceptional_finish_internal(boost::exception_ptr const& e,
    boost::unique_lock<boost::mutex>& lock) {

    exception_ = e;
    mark_finished_internal(lock);

  }

  void mark_exceptional_finish() {

    boost::unique_lock<boost::mutex> lock(this->mutex_);
    mark_exceptional_finish_internal(boost::current_exception(), lock);

  }

  void set_exception_at_thread_exit(boost::exception_ptr e) {

    boost::unique_lock<boost::mutex> lock(this->mutex_);
    if (has_value(lock)) {
      boost::throw_exception(boost::promise_already_satisfied());
    }
    exception_ = e;

    this->is_constructed_ = true;
    boost::detail::make_ready_at_thread_exit(shared_from_this());

  }

  bool has_value() const {

    boost::lock_guard<boost::mutex> lock(this->mutex_);
    return done_ && !exception_;

  }

  bool has_value(boost::unique_lock<boost::mutex>&) const {

    return done_ && !exception_;

  }

  bool has_exception() const {

    boost::lock_guard<boost::mutex> lock(this->mutex_);
    return done_ && exception_;

  }

  boost::launch launch_policy(boost::unique_lock<boost::mutex>&) const {

    return policy_;

  }

  boost::future_state::state get_state() const {

    boost::lock_guard<boost::mutex> lock(this->mutex_);
    if (!done_) {
      return boost::future_state::waiting;
    } else {
      return boost::future_state::ready;
    }

  }

  boost::exception_ptr get_exception_ptr() {

    boost::unique_lock<boost::mutex> lock(this->mutex_);
    wait_internal(lock, false);

  }

  template <typename F, typename U>
  void set_wait_callback(F f, U* u) {

    boost::lock_guard<boost::mutex> lock(this->mutex_);
    callback_ = boost::bind(f, boost::ref(*u));

  }

  virtual void execute(boost::unique_lock<boost::mutex>&) {}

private:
  shared_state_base(shared_state_base const&);
  shared_state_base& operator=(shared_state_base const&);
};
} // detail
} // boost

#endif // SHARED_STATE_BASE_IPP
