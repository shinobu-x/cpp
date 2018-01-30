/**
 * Template parameters order:
 * tempalte <
 *   typename E       // Executor type
 *   typenema F       // Function type
 *   typename S       // State type
 *   typename C       // Callback function type
 *   typenmae St      // Shared state type
 *   typename Ex      // Exception type
 *   typename... Args // Prameter pack
 * >
 * // end template parameters order
 */

#include "../include/futures.hpp"

namespace boost {

template <class T>
boost::shared_ptr<T> static_shared_from_this(T* that) {
  return boost::static_pointer_cast<T>(that->shared_from_this());
}

template <class T>
boost::shared_ptr<T const> static_shared_from_this(T const* that) {
  return boost::static_pointer_cast<T const>(that->shared_from_this());
}

class executor;

typedef boost::shared_ptr<executor> executor_ptr_type;

namespace detail {

struct relocker {

  boost::unique_lock<boost::mutex>& lock_;
  relocker(boost::unique_lock<boost::mutex>& lk) : lock_(lk) {
    lock_.unlock();
  }

  ~relocker() {
    if (!lock_.owns_lock())
      lock_.lock();
  }

  void lock() {
    if (!lock_.owns_lock())
      lock_.lock();
  }
private:
  relocker& operator=(relocker const&);
}; // relocker

/* boost::detail::shared_state_base */
struct shared_state_base :
  boost::enable_shared_from_this<shared_state_base> {
  typedef std::list<boost::condition_variable_any*> waiter_list;
  typedef waiter_list::iterator notify_when_ready_handle;
  typedef boost::shared_ptr<shared_state_base> continuation_ptr_type;
  typedef std::vector<continuation_ptr_type> continuations_type;

  boost::exception_ptr exception;
  bool done_;
  bool is_valid_;
  bool is_deferred_;
  bool is_constructed_;
  boost::launch policy_;
  mutable boost::mutex mutex_;
  boost::condition_variable waiters_;
  waiter_list external_waiters_;
  boost::function<void()> callback_;
  continuations_type continuations_;
  executor_ptr_type ex_;

  virtual void launch_continuation() {}

  shared_state_base() :
    done_(false),
    is_valid_(true),
    is_deferred_(false),
    is_constructed_(false),
    policy_(boost::launch::none),
    continuations_(),
    ex_() {}

  shared_state_base(boost::exceptional_ptr const& ex) :
    exception(ex.ptr_),
    done_(true),
    is_valid_(true),
    is_deferred_(false),
    is_constructed_(false),
    policy_(boost::launch::none),
    continuations_(),
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
#endif

    notify_when_ready_handle notify_when_ready(
      boost::condition_variable_any&  cv) {
      boost::unique_lock<boost::mutex> lock(this->mutex_);
      do_callback(lock);
      return external_waiters_.insert(external_waiters_.end(), &cv);
    }

    void unnotify_when_ready(notify_when_ready_handle it) {
      boost::lock_guard<boost::mutex> lock(this->mutex_);
      external_waiters_.erase(it);
    }

#if defined BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
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
#endif

#if defined BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
    virtual void set_continuation_ptr(continuation_ptr_type continuation,
      boost::unique_lock<boost::mutex>& lock) {
      continuations_.push_back(continuation);

      if (done_) {
        do_continuation(lock);
      }
    }
#endif

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

#if defined BOOST_THREAD_USES_DATETIME

#endif
}; // shared_state_base
} // namespace detail
} // namespace boost
