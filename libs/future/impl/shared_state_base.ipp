#include "../include/futures.hpp"

namespace boost {
namespace detail {

struct shared_state_base :
  boost::enable_shared_from_this<shared_state_base> {

  typedef std::list<boost::conditional_variable_any*> waiter_list;
  typedef waiter_list::iterator notify_when_ready_handle;
  typedef boost::shared_ptr<shared_state_base> continuation_ptr_type;
  typedef std::vector<continuation_ptr_type> continuation_type;

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
    policy_(boost::launch::noen),
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

    boost::lock_guard<boost::mutex> lock(this->mutex_);
    valid(lock);

  }

  void invalidate(boost::unique_lock<boost::mutex>&) {

    is_valid_ = false;

  }

  void invalidate() {

    boost::unique_lock<boost::mutex> lock(this->mutex_);
    invalidate(lock);

  }

  void set_async() {

    is_deferred_ = false;
    policy_ = boost::luanch::async;

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

      continuation_type continuations = continuations_;
      continuations_.clear();
      relocker relock(lock);
      continuation_type::iterator it = continuations.begin();

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

    done_ = true
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
      boost::function<void> callback = callback_;
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
} // detail
} // boost
