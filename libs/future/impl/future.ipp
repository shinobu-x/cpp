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

struct relocker;
struct shared_state_base;
template <typename T>
struct shared_state;

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

  boost::exception_ptr exception_;
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
    exception_(ex.ptr_),
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
#endif // BOOST_THREAD_PROVIDES_EXECUTORS

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
    future_status wait_until(
      const chrono::time_point<Clock, Duration>& abs_time) {
      boost::unique_lock<boost::mutex> lock(this->mutex_);

      if (is_deferred_) {
        return boost::future_status::deferred;
      }

      do_callback(lock);

      while (!done_) {
        cv_status const status = waiters_.wait_until(lock, abs_time);

        if (status == cv_status::timeout && !done_) {
          return boost::future_status::timeout;
        }
      }

      return boost::future_status::ready;
    }
#endif // BOOST_THREAD_USES_CHRONO

    void mark_exceptional_finish_internal(
      boost::exception_ptr const& e, boost::unique_lock<boost::mutex>& lock) {
      exception_ = e;
      mark_finished_internal(lock);
    }

    void mark_exceptional_finish() {
      boost::unique_lock<boost::mutex> lock(this->mutex_);
      mark_exceptional_finish_internal(boost::current_exception(), lock);
    }

    void set_exception_at_thread_exit(exception_ptr e) {
      boost::unique_lock<boost::mutex> lock(this->mutex_);

      if (has_value(lock)) {
        boost::throw_exception(promise_already_satisfied());
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

    boost::future_state::state get_state(
      boost::unique_lock<boost::mutex>&) const {
      if (!done_) {
        return boost::future_state::waiting;
      } else {
        return boost::future_state::ready;
      }
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
      return exception_;
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
}; // shared_state_base

template <typename T>
struct shared_state : boost::detail::shared_state_base {
#ifdef BOOST_THREAD_FUTURE_USES_OPTIONAL
  typedef boost::optional<T> storage_type;
#else
  typedef boost::csbl::unique_ptr<T> storage_type;
#endif // BOOST_THREAD_FUTURE_USES_OPTIONAL

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
  typedef T const& source_reference_type;
  typedef BOOST_THREAD_RV_REF(T) rvalue_source_type;
  typedef T move_dest_type;
#elif defined BOOST_THREAD_USES_MOVE
  typedef typename boost::conditional<
    boost::is_fundamental<T>::value, T, T const&>::type source_reference_type;
  typedef BOOST_THREAD_RV_REF(T) rvalue_source_type;
  typedef T move_dest_type;
#else
  typedef T& cource_reference_type;
  typedef typename boost::conditional<
    boost::thread_detail::is_convertible<
      T&,
      BOOST_THREAD_RV_REF(T)>::value,
    BOOST_THREAD_RV_REF(T), T const&>::type rvalue_source_type;
  typedef typename boost::conditional<
    boost::thread_detail::is_convertible<
      T&,
      BOOST_THREAD_RV_REF(T)>::value,
    BOOST_THREAD_RV_REF(T), T>::type move_dest_type;
#endif // BOOST_NO_CXX11_RVALUE_REFERENCES

  typedef const T& shared_future_get_result_type;
  storage_type result_;

  shared_state() : result_() {}
  shared_state(boost::exceptional_ptr const& ex) :
    boost::detail::shared_state_base(ex), result_() {}
  ~shared_state() {}

  void mark_finished_with_result_internal(source_reference_type result,
    boost::unique_lock<boost::mutex>& lock) {
#ifdef BOOST_THREAD_FUTURE_USES_OPTIONAL
    result_ = result;
#else
    result_.reset(new T(result));
#endif // BOOST_THREAD_FUTURE_USES_OPTIONAL
    this->mark_finished_internal(lock);
  }

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
  template <typename... Args>
  void mark_finished_with_result_internal(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_FWD_REF(Args) ...args) {
#ifdef BOOST_THREAD_FUTURES_USES_OPTIONAL
    result_.emplace(boost::forward<Args>(args)...);
#else
    result_.reset(new T(boost::forward<Args>(args)...));
#endif // BOOST_THREAD_FUTURES_USES_OPTIONAL
    this->mark_finished_internal(lock);
  }
#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATES

  void mark_finished_with_result(source_reference_type result) {
    boost::unique_lock<boost::mutex> lock(this->mutex_);
    this->mark_finished_with_result_internal(result, lock);
  }

  void mark_finished_with_result(rvalue_source_type result) {
    boost::unique_lock<boost::mutex> lock(this->mutex_);
#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    mark_finished_with_result_internal(boost::move(result), lock);
#else
    mark_finished_with_result_internal(
      static_cast<rvalue_source_type>(result), lock);
#endif
  }

  storage_type& get_storage(boost::unique_lock<boost::mutex>& lock) {
    wait_internal(lock);
    return result_;
  }

  virtual move_dest_type get(boost::unique_lock<boost::mutex>& lock) {
    return boost::move(*get_storage(lock));
  }

  move_dest_type get() {
    boost::unique_lock<boost::mutex> lock(this->mutex_);
    return this->get(lock);
  }

  virtual shared_future_get_result_type get_result_type(
    boost::unique_lock<boost::mutex>& lock) {
    return *get_storage(lock);
  }

  shared_future_get_result_type get_result_type() {
    boost::unique_lock<boost::mutex> lock(this->mutex);
    return get_result_type(lock);
  }

  void set_value_at_thread_exit(source_reference_type result) {
    boost::unique_lock<boost::mutex> lock(this->mutex_);

    if (this->has_value(lock)) {
      boost::throw_exception(boost::promise_already_satisfied());
    }
#ifdef BOOST_THREAD_FUTURES_USES_OPTIONAL
    result_ = result;
#else
    result_.reset(new T(result));
#endif // BOOST_THREAD_FUTURES_USES_OPTIONAL
    this->is_constructed_ = true;

    boost::detail::make_ready_at_thread_exit(shared_from_this());
  }

  void set_value_at_thread_exit(rvalue_source_type result) {
    boost::unique_lock<boost::mutex> lock(this->mutex_);

    if (this->has_value(lock)) {
      boost::throw_exception(boost::promise_already_satisfied());
    }
#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
#ifdef BOOST_THREAD_FUTURE_USES_OPTIONAL
    result_ = boost::move(result);
#else
    result_.reset(new T(static_cast<rvalue_source_type>(result)));
#endif // BOOST_THREAD_FUTURE_USES_OPTIONAL
#endif // BOOST_NO_CXX11_RVALUE_REFERENCES
    this->is_constructed_ = true;
    boost::detail::make_ready_at_thread_exit(shared_from_this());
  }

private:
  shared_state(shared_state const&);
  shared_state& operator=(shared_state const&);
}; // shared_state

template <typename T>
struct shared_state<T&> : boost::detail::shared_state_base {
  typedef T* storage_type;
  typedef T& source_reference_type;
  typedef T& move_dest_type;
  typedef T& shared_future_get_result_type;

  T* result_;

  shared_state() : result_(0) {}
  shared_state(boost::exceptional_ptr const& ex) :
    boost::detail::shared_state_base(ex), result_(0) {}
  ~shared_state() {}

  void mark_finished_with_result_internal(source_reference_type result,
    boost::unique_lock<boost::mutex>& lock) {
    result_ = result;
    mark_finished_internal(lock);
  }

  void mark_finished_with_result(source_reference_type result) {
    boost::unique_lock<boost::mutex> lock(this->mutex_);
    mark_finished_with_result_internal(result, lock);
  }

  virtual T& get(boost::unique_lock<boost::mutex>& lock) {
    wait_internal(lock);
    return *result_;
  }

  T& get() {
    boost::unique_lock<boost::mutex> lock(this->mutex_);
    return get(lock);
  }

  virtual T& get_result_type(boost::unique_lock<boost::mutex>& lock) {
    wait_internal(lock);
    return *result_;
  }

  T* get_result_type() {
    boost::unique_lock<boost::mutex> lock(this->mutex_);
    return get_result_type(lock);
  }

  void set_value_at_thread_exit(T& result) {
    boost::unique_lock<boost::mutex> lock(this->mutex_);

    if (this->has_value(lock)) {
      boost::throw_exception(boost::promise_already_satisfied());
    }

    result_ = result;
    this->is_constructed_ = true;
    boost::detail::make_ready_at_thread_exit(shared_from_this());
  }

private:
  shared_state(shared_state const&);
  shared_state& operator=(shared_state const&);
}; // shared_state

template <>
struct shared_state<void> :
  boost::detail::shared_state_base {
  typedef void shared_future_get_result_type;
  typedef void move_dest_type;

  shared_state() {}
  shared_state(boost::exceptional_ptr const& ex) :
    boost::detail::shared_state_base(ex) {}

  void mark_finished_with_result_internal(
    boost::unique_lock<boost::mutex>& lock) {
    mark_finished_internal(lock);
  }

  void mark_finished_with_result() {
    boost::unique_lock<boost::mutex> lock(this->mutex_);
    mark_finished_with_result_internal(lock);
  }

  virtual void get(boost::unique_lock<boost::mutex>& lock) {
    this->wait_internal(lock);
  }

  void get() {
    boost::unique_lock<boost::mutex> lock(this->mutex_);
    this->get(lock);
  }

  virtual void get_result_type(boost::unique_lock<boost::mutex>& lock) {
    this->wait_internal(lock);
  }

  void get_result_type() {
    boost::unique_lock<boost::mutex> lock(this->mutex_);
    this->get_result_type(lock);
  }

  void set_value_at_thread_exit() {
    boost::unique_lock<boost::mutex> lock(this->mutex_);

    if (this->has_value(lock)) {
      boost::throw_exception(boost::promise_already_satisfied());
    }

    this->is_constructed_ = true;
    boost::detail::make_ready_at_thread_exit(shared_from_this());
  }

private:
  shared_state(shared_state const&);
  shared_state& operator=(shared_state const&);
}; // shared_state

template <typename S>
struct future_async_shared_state_base :
  boost::detail::shared_state<S> {
  typedef boost::detail::shared_state<S> base_type;

protected:
#ifdef BOOST_THREAD_FUTURE_BLOCKING
  boost::thread thr_;
  void join() {
    if (boost::this_thread::get_id() == thr_.get_id()) {
      thr_.detach();
      return;
    }

    if (thr_.joinable()) {
      thr_.join();
    }
  }
#endif // BOOST_THREAD_FUTURE_BLOCKING
public:
  future_async_shared_state_base() {
    this->set_async();
  }

  ~future_async_shared_state_base() {
#ifdef BOOST_THREAD_FUTURE_BLOCKING
    join();
#endif // BOOST_THREAD_FUTURE_BLOCKING
  }

  virtual void wait(boost::unique_lock<boost::mutex>& lock, bool rethrow) {
#ifdef BOOST_THREAD_FUTURE_BLOCKING
    {
      relocker relock(lock);
      join();
    }
#endif // BOOST_THREAD_FUTURE_BLOCKING
    this->base_type::wait(lock, rethrow);
  }
}; // future_async_shared_state_base
} // namespace detail
} // namespace boost
