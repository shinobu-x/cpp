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
  typedef T& source_reference_type;
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

  void mark_finished_with_result_internal(rvalue_source_type result,
    boost::unique_lock<boost::mutex>& lock) {
#ifdef BOOST_THREAD_FUTURE_USES_OPTIONAL
    result_ = boost::move(result);
#elif !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
    result_.reset(new T(boost::move(result)));
#else
    result_.reset(new T(static_cast<rvalue_source_type>(result)));
#endif // BOOST_NO_CXX11_RVALUE_REFERENCES
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

/* future_async_shared_state */
template <typename S, typename F>
struct future_async_shared_state :
  boost::detail::future_async_shared_state_base<S> {
  future_async_shared_state() {}

  void init(BOOST_THREAD_FWD_REF(F) f) {
#ifdef BOOST_THREAD_FUTURE_BLOCKING
    this->thr_ = boost::thread(&future_async_shared_state::run,
      static_shared_from_this(this), boost::forward<F>(f));
#else
    boost::thread(&future_async_shared_state::run,
      static_shared_from_this(this), boost::forward<F>(f)).detach();
#endif // BOOST_THREAD_FUTURE_BLOCKING
  }

  static void run(boost::shared_ptr<future_async_shared_state> that,
    BOOST_THREAD_FWD_REF(F) f) {
    try {
      that->mark_finished_with_result(f());
    } catch (...) {
      that->mark_exceptional_finish();
    }
  }
}; // future_async_shared_state

template <typename F>
struct future_async_shared_state<void, F> :
  public boost::detail::future_async_shared_state_base<void> {
  void init(BOOST_THREAD_FWD_REF(F) f) {
#ifdef BOOST_THREAD_FUTURE_BLOCKING
    this->thr_ = boost::thread(&future_async_shared_state::run,
      static_shared_from_this(this), boost::move(f));
#else
    boost::thread(&future_async_shared_state::run,
      static_shared_from_this(this), boost::move(f)).detach();
#endif // BOOST_THREAD_FUTURE_BLOCKING
  }

  static void run(boost::shared_ptr<future_async_shared_state> that,
    BOOST_THREAD_FWD_REF(F) f) {
    try {
      f();
      that->mark_finished_with_result();
    } catch (...) {
      that->mark_exceptional_finish();
    }
  }
}; // future_async_shared_state

template <typename R, typename F>
struct future_async_shared_state<R&, F> :
  boost::detail::future_async_shared_state_base<R&> {
  void init(BOOST_THREAD_FWD_REF(F) f) {
#ifdef BOOST_THREAD_FUTURE_BLOCKING
    this->thr_ = boost::thread(&future_async_shared_state::run,
      static_shared_from_this(this), boost::move(f));
#else
    boost::thread(&future_async_shared_state::run,
      static_shared_from_this(this), boost::move(f)).detach();
#endif // BOOST_THREAD_FUTURE_BLOCKING
  }

  static void run(boost::shared_ptr<future_async_shared_state> that,
    BOOST_THREAD_FWD_REF(F) f) {
    try {
      that->mark_finished_with_result(f());
    } catch (...) {
      that->mark_exceptional_finish();
    }
  }
}; // future_async_shared_state

/* future_deferred_shared_state */
template <typename R, typename F>
struct future_deferred_shared_state :
  boost::detail::shared_state<R> {
  typedef boost::detail::shared_state<R> base_type;
  F f_;

  explicit future_deferred_shared_state(BOOST_THREAD_FWD_REF(F) f) :
    f_(boost::move(f)) {
    this->set_deferred();
  }

  virtual void execute(boost::unique_lock<boost::mutex>& lock) {
    try {
      F f(boost::move(f_));
      relocker relock(lock);
      R r = f();
      relock.lock();
      this->mark_finished_with_result_internal(boost::move(r), lock);
    } catch (...) {
      this->mark_exceptional_finish_internal(boost::current_exception(), lock);
    }
  }
}; // future_deferred_shared_state

template <typename R, typename F>
struct future_deferred_shared_state<R&, F> :
  boost::detail::shared_state<R&> {
  typedef boost::detail::shared_state<R&> base_type;
  F f_;

  explicit future_deferred_shared_state(BOOST_THREAD_FWD_REF(F) f) :
    f_(boost::move(f)) {
    this->set_deferred();
  }

  virtual void execute(boost::unique_lock<boost::mutex>& lock) {
    try {
      this->mark_finished_with_result_internal(f_(), lock);
    } catch (...) {
      this->mark_exceptional_finish_internal(boost::current_exception(), lock);
    }
  }
}; // future_deferred_shared_state

template <typename F>
struct future_deferred_shared_state<void, F> :
  boost::detail::shared_state<void> {
  typedef boost::detail::shared_state<void> base_type;
  F f_;

  explicit future_deferred_shared_state(BOOST_THREAD_FWD_REF(F) f) :
    f_(boost::move(f)) {
    this->set_deferred();
  }

  virtual void execute(boost::unique_lock<boost::mutex>& lock) {
    try {
      F f = boost::move(f_);
      relocker relock(lock);
      f();
      relock.lock();
      this->mark_finished_with_result_internal(lock);
    } catch (...) {
      this->mark_exceptional_finish_internal(boost::current_exception(), lock);
    }
  }
}; // future_deferred_shared_state

class future_waiter {
public:
  typedef std::vector<int>::size_type count_type;
private:
  struct registered_waiter {
    boost::shared_ptr<boost::detail::shared_state_base> future_;
    boost::detail::shared_state_base::notify_when_ready_handle handle_;
    count_type index_;

    registered_waiter(boost::shared_ptr<
        boost::detail::shared_state_base> const& future,
      boost::detail::shared_state_base::notify_when_ready_handle handle,
      count_type index) :
      future_(future), handle_(handle), index_(index) {}
  }; // registered_waiter

  struct all_futures_lock {
#ifdef _MANAGED
    typedef std::ptrdiff_t count_type_portable;
#else
    typedef count_type count_type_portable;
#endif // _MANAGED
    count_type_portable count_;
    boost::scoped_array<boost::unique_lock<boost::mutex> > locks_;

    all_futures_lock(std::vector<registered_waiter>& futures) :
      count_(futures.size()),
      locks_(new boost::unique_lock<boost::mutex>[count_]) {
      for (count_type_portable i = 0; i < count_; ++i) {
        locks_[i] = BOOST_THREAD_MAKE_RV_REF(
          boost::unique_lock<boost::mutex>(futures[i].future_->mutex_));
      }
    }

    void lock() {
      boost::lock(locks_.get(), locks_.get() + count_);
    }

    void unlock() {
      for (count_type_portable i = 0; i < count_; ++i) {
        locks_[i].unlock();
      }
    }
  }; // all_futures_lock

  boost::condition_variable_any cv_;
  std::vector<registered_waiter> futures_;
  count_type future_count_;

public:
  future_waiter() : future_count_(0) {}

  template <typename F>
  void add(F& f) {
    if (f.future_) {
      registered_waiter waiter_(f.future_,
        f.future_->notify_when_ready(cv_), future_count_);

      try {
        futures_.push_back(waiter_);
      } catch (...) {
        f.future_->unnotify_when_ready(waiter_.handle_);
        throw;
      }
    }

    ++future_count_;
  }

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATE
  template <typename F, typename... Fs>
  void add(F& f, Fs& ...fs) {
    add(f);
    add(fs...);
  }
#endif

  count_type wait() {
    all_futures_lock lock(futures_);

    for (;;) {
      for (count_type i = 0; i < futures_.size(); ++i) {
        if (futures_[i].future_->done_) {
          return futures_[i].index_;
        }
      }

      cv_.wait(lock);
    }
  }

  ~future_waiter() {
    for (count_type i = 0; i < futures_.size(); ++i) {
      futures_[i].future_->unnotify_when_ready(futures_[i].handle_);
    }
  }
}; // future_waiter
} // detail

template <typename R>
class BOOST_THREAD_FUTURE;
template <typename R>
class shared_future;
template <typename R>
class promise;
template <typename R>
class packaged_task;

template <typename T>
struct is_future_type<BOOST_THREAD_FUTURE<T> > : true_type {};

template <typename T>
struct is_future_type<shared_future<T> > : true_type {};

#ifdef BOOST_NO_CXX11_VARIADIC_TEMPLATE
template <typename F1, typename F2>
typename boost::enable_if<
  boost::is_future_type<F1>,
  typename boost::detail::future_waiter::count_type
>::type wait_for_any(F1& f1, F2& f2) {
  boost::detail::future_waiter waiter;
  waiter.add(f1);
  waiter.add(f2);
  return waiter.wait();
} // wait_for_any

template <typename F1, typename F2, typename F3>
typename boost::detail::future_waiter::count_type wait_for_any(
  F1& f1, F2& f2, F3& f3) {
  boost::detail::future_waiter waiter;
  waiter.add(f1);
  waiter.add(f2);
  waiter.add(f3);
  return waiter.wait();
} // wait_for_any

template <typename F1, typename F2, typename F3, typename F4>
typename boost::detail::future_waiter::count_type wait_for_any(
  F1& f1, F2& f2, F3& f3, F4& f4) {
  boost::detail::future_waiter waiter;
  waiter.add(f1);
  waiter.add(f2);
  waiter.add(f3);
  waiter.add(f4);
  return waiter.wait();
} // wait_for_any

template <typename F1, typename F2, typename F3, typename F4, typename F5>
typename boost::detail::future_waiter::count_type wait_for_any(
  F1& f1, F2& f2, F3& f3, F4& f4, F5& f5) {
  boost::detail::future_waiter waiter;
  waiter.add(f1);
  waiter.add(f2);
  waiter.add(f3);
  waiter.add(f4);
  waiter.add(f5);
  return waiter.wait();
} // wait_for_any
#endif

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATE
template <typename F, typename... Fs>
typename boost::enable_if<
  boost::is_future_type<F>,
  typename boost::detail::future_waiter::count_type>::type wait_for_any(
  F& f, Fs ...fs) {
  boost::detail::future_waiter waiter;
  waiter.add(f, fs...);
  return waiter.wait();
}
#endif

namespace detail {

class base_future {
public:  
}; // base_future

template <typename R>
class basic_future : public base_future {
protected:
public:
  typedef boost::shared_ptr<boost::detail::shared_state<R> > future_ptr;
  typedef typename boost::detail::shared_state<R>::move_dest_type
    move_dest_type;

  static future_ptr make_exceptional_future_ptr(
    boost::exceptional_ptr const& ex) {
    return future_ptr(new boost::detail::shared_state<R>(ex));
  }

  future_ptr future_;
  basic_future(future_ptr future) : future_(future) {}
  typedef boost::future_state::state state;

  BOOST_THREAD_MOVABLE_ONLY(basic_future) basic_future() : future_() {}

  basic_future(boost::exceptional_ptr const& ex) :
    future_(make_exceptional_future_ptr(ex)) {}

  ~basic_future() {}

  basic_future(BOOST_THREAD_RV_REF(basic_future) that) BOOST_NOEXCEPT :
    future_(BOOST_THREAD_RV(that).future_) {
    BOOST_THREAD_RV(that).future_.reset();
  }

  basic_future& operator=(
    BOOST_THREAD_RV_REF(basic_future) that) BOOST_NOEXCEPT {
    future_ = BOOST_THREAD_RV(that).future_;
    BOOST_THREAD_RV(that).future_.reset();
    return *this;
  }

  void swap(basic_future& that) BOOST_NOEXCEPT {
    future_.swap(that.future_);
  }

  state get_state(boost::unique_lock<boost::mutex>& lock) const {
    if (!future_) {
      return boost::future_state::uninitialized;
    }
    return future_->get_state(lock);
  }

  state get_state() const {
    if (!future_) {
      return boost::future_state::uninitialized;
    }
    return future_->get_state();
  }

  bool is_ready() const {
    return get_state() == boost::future_state::ready;
  }

  bool is_ready(boost::unique_lock<boost::mutex>& lock) const {
    return get_state(lock) == boost::future_state::ready;
  }

  bool has_exception() const {
    return future_ && future_->has_exception();
  }

  bool has_value() const {
    return future_ && future_->has_value();
  }

  boost::launch launch_policy(boost::unique_lock<boost::mutex>& lock) const {
    if (future_) {
      return future_->launch_policy(lock);
    } else {
      return boost::launch(boost::launch::none);
    }
  }

  boost::launch launch_policy() const {
    if (future_) {
      boost::unique_lock<boost::mutex> lock(this->future_->mutex);
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

  bool valid() const BOOST_NOEXCEPT {
    return future_ != 0 && future_->valid();
  }

  void wait() const {
    if (!future_) {
      boost::throw_exception(boost::future_uninitialized());
    }
    future_->wait(false);
  }

  typedef boost::detail::shared_state_base::notify_when_ready_handle
    notify_when_ready_handle;

  boost::mutex& mutex() {
    if (!future_) {
      boost::throw_exception(boost::future_uninitialized());
    }
    return future_->mutex;
  }

  notify_when_ready_handle notify_when_ready(
    boost::condition_variable_any& cv) {
    if (!future_) {
      boost::throw_exception(boost::future_uninitialized());
    }
    return future_->notify_whe_ready(cv);
  }

  void unnotify_when_ready(notify_when_ready_handle h) {
    if (!future_) {
      boost::throw_exception(boost::future_uninitialized());
    }
    return future_->unnotify_when_ready(h);
  }

#ifdef BOOST_THREAD_USES_DATE
template <typename Duration>
bool timed_wait(Duration const& real_time) const {
  return timed_wait_until(boost::get_system_time() + real_time);
}

bool timed_wait_until(boost::system_time const& abs_time) const {
  if (!future_) {
    boost::throw_exception(boost::future_uninitialized());
  }
  return future_->timed_wait_until(abs_time);
}
#endif
#ifdef BOOST_THREAD_USES_CHRONO
template <typename Rep, typename Period>
boost::future_status wait_for(
  const boost::chrono::duration<Rep, Period>& real_time) const {
  return wait_until(boost::chrono::steady_clock::now() + real_time);
}

template <typename Clock, typename Duration>
boost::future_status wait_until(
  const boost::chrono::time_point<Clock, Duration>& abs_time) const {
  if (!future_) {
    boost::throw_exception(boost::future_uninitialized());
  }
  return future_->wait_until(abs_time);
}
#endif
}; // basic_future
} // detail

BOOST_THREAD_DCL_MOVABLE_BEG(R)
boost::detail::basic_future<R>
BOOST_THREAD_DCL_MOVABLE_END

namespace detail {

#if (!defined _MSC_VER || _MSC_VER >= 1400)
template <typename R, typename F>
BOOST_THREAD_FUTURE<R> make_future_async_shared_state(
  BOOST_THREAD_FWD_REF(F) f);

template <typename R, typename F>
BOOST_THREAD_FUTURE<R> make_future_deferred_shared_state(
  BOOST_THREAD_FWD_REF(F) f);
#endif // (!defined _MSC_VER || _MSC_VER >= 1400)

#ifdef BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
template <typename F, typename R, typename C>
struct future_async_continuation_shared_state;

template <typename F, typename R, typename C>
struct future_deferred_continuation_shared_state;

template <typename F, typename R, typename C>
BOOST_THREAD_FUTURE<R> make_future_async_continuation_shared_state(
  boost::unique_lock<boost::mutex>& lock,
  BOOST_THREAD_RV_REF(F) f,
  BOOST_THREAD_FWD_REF(C) c);

template <typename F, typename R, typename C>
BOOST_THREAD_FUTURE<R> make_future_sync_continuation_shared_state(
  boost::unique_lock<boost::mutex>& lock,
  BOOST_THREAD_RV_REF(F) f,
  BOOST_THREAD_FWD_REF(C) c);

template <typename F, typename R, typename C>
BOOST_THREAD_FUTURE<R> make_future_deferred_continuation_shared_state(
  boost::unique_lock<boost::mutex>& lock,
  BOOST_THREAD_RV_REF(F) f,
  BOOST_THREAD_FWD_REF(C) c);

template <typename F, typename R, typename C>
BOOST_THREAD_FUTURE<R> make_shared_future_async_continuation_shared_state(
  boost::unique_lock<boost::mutex>& lock,
  F f,
  BOOST_THREAD_FWD_REF(C) c);

template <typename F, typename R, typename C>
BOOST_THREAD_FUTURE<R> make_shared_future_sync_continuation_shared_state(
  boost::unique_lock<boost::mutex>& lock,
  F f,
  BOOST_THREAD_FWD_REF(C) c);

template <typename F, typename R, typename C>
BOOST_THREAD_FUTURE<R> make_shared_future_deferred_continuation_shared_state(
  boost::unique_lock<boost::mutex>& lock,
  F f,
  BOOST_THREAD_FWD_REF(C) c);

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
template <typename R, typename C, typename E>
BOOST_THREAD_FUTURE<R> make_future_executor_shared_state(
  E& e,
  BOOST_THREAD_RV_REF(C) c);

template <typename E, typename F, typename R, typename C>
BOOST_THREAD_FUTURE<R> make_future_executor_continuation_shared_state(
  E& e,
  BOOST_THREAD_RV_REF(F) f,
  BOOST_THREAD_FWD_REF(C) c);

template <typename E, typename F, typename R, typename C>
BOOST_THREAD_FUTURE<R> make_shared_future_executor_continuation_shared_state(
  E& e,
  F f,
  BOOST_THREAD_FWD_REF(C) c);
#endif // BOOST_THREAD_PROVIDES_EXECUTORS
#endif // BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION

#ifdef BOOST_THREAD_PROVIDES_FUTURE_UNWRAP
template <typename F, typename R>
struct future_unwrap_shared_state;

template <typename F, typename R>
inline BOOST_THREAD_FUTURE<R> make_future_unwrap_shared_state(
  boost::unique_lock<boost::mutex>& lock,
  BOOST_THREAD_RV_REF(F) f);
#endif
} // detail

#ifdef BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY
template <typename InputIter>
typename boost::disable_if<
  boost::is_future_type<InputIter>,
  BOOST_THREAD_FUTURE<
    boost::csbl::vector<typename InputIter::value_type> > >::type when_all(
  InputIter first, InputIter last);
#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATE
template <typename T, typename ...Ts>
BOOST_THREAD_FUTURE<boost::csbl::tuple<
  typename boost::decay<T>::type,
  typename boost::decay<Ts>::type...> > when_all(
  BOOST_THREAD_FWD_REF(T) f,
  BOOST_THREAD_FWD_REF(Ts) ...futures);
#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATE
template <typename InputIter>
typename boost::disable_if<
  boost::is_future_type<InputIter>,
  BOOST_THREAD_FUTURE<
    boost::csbl::vector<typename InputIter::value_type> > > when_any(
  InputIter first, InputIter last);

inline BOOST_THREAD_FUTURE<boost::csbl::tuple<> > when_any();

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATE
template <typename T, typename... Ts>
BOOST_THREAD_FUTURE<boost::csbl::tuple<
  typename boost::decay<T>::type,
  typename boost::decay<Ts>::type...> > when_any(
  BOOST_THREAD_FWD_REF(T) f,
  BOOST_THREAD_FWD_REF(Ts) ...futures);
#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATE
#endif // BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY

template <typename T>
class BOOST_THREAD_FUTURE : public boost::detail::basic_future<T> {
// private
  typedef boost::detail::basic_future<T> base_type;
  typedef typename base_type::future_ptr future_ptr;

  friend class shared_future<T>;
  friend class promise<T>;

#ifdef BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
  template <typename, typename, typename>
  friend struct boost::detail::future_async_continuation_shared_state;

  template <typename, typename, typename>
  friend struct boost::detail::future_deferred_continuation_shared_state;

  // future
  template <typename F, typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
  boost::detail::make_future_async_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
  boost::detail::make_future_sync_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
  boost::detail::make_future_deferred_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c);

  // shared_future
  template <typename F, typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
  boost::detail::make_shared_future_async_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lokc,
    F f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
  boost::detail::make_shared_future_sync_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    F f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
  boost::detail::make_shared_future_deferred_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    F f,
    BOOST_THREAD_FWD_REF(C) c);

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
  template <typename R, typename C, typename E>
  friend BOOST_THREAD_FUTURE<R>
  boost::detail::make_future_executor_shared_state(
    E& e,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename E, typename F, typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
  boost::detail::make_future_executor_continuation_shared_state(
    E& e,
    boost::unique_lock<boost::mutex>& lock,
    F f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename E, typename F, typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
  boost::detail::make_shared_future_executor_continuation_shared_state(
    E& e,
    boost::unique_lock<boost::mutex>& lock,
    F f,
    BOOST_THREAD_FWD_REF(C) c);
#endif // BOOST_THREAD_PROVIDES_EXECUTORS
#endif // BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION

#ifdef BOOST_THREAD_PROVIDES_FUTURE_UNWRAP
  template <typename F, typename R>
  friend struct boost::detail::future_unwrap_shared_state;

  template <typename F, typename R>
  friend BOOST_THREAD_FUTURE<R>
  boost::detail::make_future_unwrap_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f);
#endif // BOOST_THREAD_PROVIDES_FUTURE_UNWRAP

#ifdef BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY
  template <typename InputIter>
  friend typename boost::disable_if<
    boost::is_future_type<InputIter>,
    BOOST_THREAD_FUTURE<boost::csbl::vector<typename InputIter::value_type> >
  >::type when_all(InputIter first, InputIter last);

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
  template <typename U, typename... Us>
  friend BOOST_THREAD_FUTURE<
    boost::csbl::tuple<typename boost::decay<U>::type,
    typename boost::decay<Us>::type...> > when_all(
      BOOST_THREAD_FWD_REF(U) f,
      BOOST_THREAD_FWD_REF(Us) ...futures);
#endif

  template <typename InputIter>
  friend typename boost::disable_if<
    boost::is_future_type<InputIter>,
    BOOST_THREAD_FUTURE<boost::csbl::vector<typename InputIter::value_type> >
  >::type when_any(InputIter first, InputIter last);

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
  template <typename U, typename... Us>
  friend BOOST_THREAD_FUTURE<
    boost::csbl::tuple<typename boost::decay<U>::type,
    typename boost::decay<Us>::type...> > when_any(
      BOOST_THREAD_FWD_REF(U) f,
      BOOST_THREAD_FWD_REF(Us) ...futures);
#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATES
#endif // BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY

#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
  template <class>
  friend class packaged_task;
#else
  friend class packaged_task<T>;
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
  friend class boost::detail::future_waiter;

  template <typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
    boost::detail::make_future_async_shared_state(
      BOOST_THREAD_FWD_REF(C) c);

  template <typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
    boost::detail::make_future_deferred_shared_state(
      BOOST_THREAD_FWD_REF(C) c);

  // boost::detail::basic_future<T> base_type
  typedef typename base_type::move_dest_type move_dest_type;

  BOOST_THREAD_FUTURE(future_ptr future) : base_type(future) {}

public:
  BOOST_THREAD_MOVABLE_ONLY(BOOST_THREAD_FUTURE)
  typedef boost::future_state::state state;
  typedef T value_type;

  BOOST_CONSTEXPR BOOST_THREAD_FUTURE() {}
  BOOST_THREAD_FUTURE(boost::exceptional_ptr const& ex) : base_type(ex) {}
  ~BOOST_THREAD_FUTURE() {}

  BOOST_THREAD_FUTURE(
    BOOST_THREAD_RV_REF(BOOST_THREAD_FUTURE) that) BOOST_NOEXCEPT :
     base_type(boost::move(static_cast<base_type&>(
       BOOST_THREAD_RV(that)))) {}

#ifdef BOOST_PROVIDES_FUTURE_UNWRAP
  inline explicit BOOST_THREAD_FUTURE(
    BOOST_THREAD_RV_REF(
      BOOST_THREAD_FUTURE<BOOST_THREAD_FUTURE<R> >) that);
#endif // BOOST_PROVIDES_FUTURE_UNWRAP

  explicit BOOST_THREAD_FUTURE(
    BOOST_THREAD_RV_REF(shared_future<T>) that) :
      base_type(boost::move(
        static_cast<base_type&>(BOOST_THREAD_RV(that)))) {}

  BOOST_THREAD_FUTURE& operator=(
    BOOST_THREAD_RV_REF(BOOST_THREAD_FUTURE) that) BOOST_NOEXCEPT {
      this->base_type::operator=(
        boost::move(
          static_cast<base_type&>(BOOST_THREAD_RV(that))));
    return *this;
  }

  shared_future<T> share() {
    return shared_future<T>(boost::move(*this));
  }

// private
  void set_async() {
    this->future_->set_async();
  }

// private
  void set_deferred() {
    this->future_->set_deferred();
  }

  bool run_if_is_deferred() {
    return this->future_->run_if_is_deferred();
  }

  bool run_if_is_deferred_or_ready() {
    return this->future_->run_if_is_deferred_or_ready();
  }

  move_dest_type get() {
    if (this->future_.get() == 0) {
      boost::throw_exception(boost::future_uninitialized());
    }

    boost::unique_lock<boost::mutex> lock(this->future_->mutex_);

    if (!this->future_->valid(lock)) {
      boost::throw_exception(boost::future_uninitialized());
    }

#ifdef BOOST_THREAD_PROVIDES_FUTURE_INVALID_AFTER_GET
    this->future_->invalidate(lock);
#endif // BOOST_THREAD_PROVIDES_FUTURE_INVALID_AFTER_GET

    return this->future_->get(lock);
  }

  template <typename T2>
  typename boost::disable_if<
    boost::is_void<T2>,
    move_dest_type>::type get_or(BOOST_THREAD_RV_REF(T2) v) {
    if (this->future_.get() == 0) {
      boost::throw_exception(boost::future_uninitialized());
    }

    boost::unique_lock<boost::mutex> lock(this->future_->mutex_);

    if (!this->future_->valid(lock)) {
      boost::throw_exception(boost::future_uninitialized());
    }

    this->future_->wait(lock, false);

#ifdef BOOST_THREAD_PROVIDES_FUTURE_INVALID_AFTER_GET
    this->future_->invalidate(lock);
#endif  // BOOST_THREAD_PROVIDES_FUTURE_INVALID_AFTER_GET

    if (this->future_->has_value(lock)) {
      return this->future_->get(lock);
    } else {
      return boost::move(v);
    }
  }

  template <typename T2>
  typename boost::disable_if<
    boost::is_void<T2>,
    move_dest_type>::type get_or(T2 const& v) {
    if (this->future_.get() == 0) {
      boost::throw_exception(boost::future_uninitialized());
    }

    boost::unique_lock<boost::mutex> lock(this->future_->mutex_);

    if (!this->future_->valid(lock)) {
      boost::throw_exception(boost::future_uninitialized());
    }

    this->future_->wait(lock, false);

#ifdef BOOST_THREAD_PROVIDES_FUTURE_INVALID_AFTER_GET
    this->future_->invalidate(lock);
#endif  // BOOST_THREAD_PROVIDES_FUTURE_INVALID_AFTER_GET

    if (this->future_->has_valid(lock)) {
      return this->future_->get(lock);
    } else {
      return v;
    }
  }

#ifdef BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
  template <typename F>
  inline BOOST_THREAD_FUTURE<
    typename boost::result_of<F(BOOST_THREAD_FUTURE)>::type> then(
      BOOST_THREAD_FWD_REF(F) f);

  template <typename F>
  inline BOOST_THREAD_FUTURE<
    typename boost::result_of<F(BOOST_THREAD_FUTURE)>::type> then(
      boost::launch policy,
      BOOST_THREAD_FWD_REF(F) f);

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
  template <typename E, typename F>
  inline BOOST_THREAD_FUTURE<
    typename boost::result_of<F(BOOST_THREAD_FUTURE)>::type> then(
      E& e,
      BOOST_THREAD_FWD_REF(F) f);
#endif // BOOST_THREAD_PROVIDES_EXECUTORS

  template <typename T2>
  inline typename boost::disable_if<
    boost::is_void<T2>,
    BOOST_THREAD_FUTURE<T> >::type fallback_to(
      BOOST_THREAD_RV_REF(T2) v);

  template <typename T2>
  inline typename boost::disable_if<
    boost::is_void<T2>,
    BOOST_THREAD_FUTURE<T> >::type fallback_to(
      T2 const& v);
#endif // BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
}; // BOOST_THREAD_FUTURE

BOOST_THREAD_DCL_MOVABLE_BEG(T)
boost::BOOST_THREAD_FUTURE<T>
BOOST_THREAD_DCL_MOVABLE_END

template <typename T2>
class BOOST_THREAD_FUTURE<BOOST_THREAD_FUTURE<T2> > :
  public boost::detail::basic_future<BOOST_THREAD_FUTURE<T2> > {
  typedef BOOST_THREAD_FUTURE<T2> T;

// private
  typedef boost::detail::basic_future<T> base_type;
  typedef typename base_type::future_ptr future_ptr;

  friend class boost::shared_future<T>;
  friend class boost::promise<T>;

#ifdef BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
  template <typename, typename, typename>
  friend struct boost::detail::future_async_continuation_shared_state;

  template <typename, typename, typename>
  friend struct boost::detail::future_deferred_continuation_shared_state;

  /* future */
  template <typename F, typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
    boost::detail::make_future_async_continuation_shared_state(
      boost::unique_lock<boost::mutex>& lock,
      BOOST_THREAD_RV_REF(F) f,
      BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
    boost::detail::make_future_sync_continuation_shared_state(
      boost::unique_lock<boost::mutex>& lock,
      BOOST_THREAD_RV_REF(F) f,
      BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
    boost::detail::make_future_deferred_continuation_shared_state(
      boost::unique_lock<boost::mutex>& lock,
      BOOST_THREAD_RV_REF(F) f,
      BOOST_THREAD_FWD_REF(C) c);

  /* shared future */
  template <typename F, typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
    boost::detail::make_shared_future_async_continuation_shared_state(
      boost::unique_lock<boost::mutex>& lock,
      F f,
      BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
    boost::detail::make_shared_future_sync_continuation_shared_state(
      boost::unique_lock<boost::mutex>& lock,
      F f,
      BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
    boost::detail::make_shared_future_deferred_continuation_shared_state(
      boost::unique_lock<boost::mutex>& lock,
      F f,
      BOOST_THREAD_FWD_REF(C) c);

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
  template <typename R, typename F, typename E>
  friend BOOST_THREAD_FUTURE<R>
    boost::detail::make_future_executor_shared_state(
      E& e,
      BOOST_THREAD_FWD_REF(F) f);

  template <typename E, typename F, typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
    boost::detail::make_future_executor_continuation_shared_state(
      E& e,
      boost::unique_lock<boost::mutex>& lock,
      F f,
      BOOST_THREAD_FWD_REF(C) c);

  template <typename E, typename F, typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
    boost::detail::make_shared_future_executor_continuation_shared_state(
      E& e,
      boost::unique_lock<boost::mutex>& lock,
      F f,
      BOOST_THREAD_FWD_REF(C) c);
#endif // BOOST_THREAD_PROVIDES_EXECUTORS
#endif // BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION

#ifdef BOOST_THREAD_PROVIDES_FUTURE_UNWRAP
  template <typename F, typename R>
  friend struct boost::detail::future_unwrap_shared_state;

  template <typename F, typename R>
  friend BOOST_THREAD_FUTURE<R>
    boost::detail::make_future_unwrap_shared_state(
      boost::unique_lock<boost::mutex>& lock,
      BOOST_THREAD_RV_REF(F) f);
#endif // BOOST_THREAD_PROVIDES_FUTURE_UNWRAP

#ifdef BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY
  template <typename InputIter>
  friend typename boost::disable_if<
    boost::is_future_type<InputIter>,
    BOOST_THREAD_FUTURE<
      boost::csbl::vector<typename InputIter::value_type> > >::type
        when_all(InputIter first, InputIter last);

  friend inline BOOST_THREAD_FUTURE<boost::csbl::tuple<> > when_all();

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
  template <typename T, typename... Ts>
  friend BOOST_THREAD_FUTURE<
    boost::csbl::tuple<
      typename boost::decay<T>::type,
      typename boost::decay<Ts>::type...> >
        when_all(
          BOOST_THREAD_FWD_REF(T) f,
          BOOST_THREAD_FWD_REF(Ts) ...futures);
#endif //BOOST_NO_CXX11_VARIADIC_TEMPLATES

  template <typename InputIter>
  friend typename boost::disable_if<
    boost::is_future_type<InputIter>,
    BOOST_THREAD_FUTURE<
      boost::csbl::vector<typename InputIter::value_type> > >::type
        when_any(InputIter first, InputIter last);

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
  template <typename T, typename... Ts>
  friend BOOST_THREAD_FUTURE<
    boost::csbl::tuple<
      typename boost::decay<T>::type,
      typename boost::decay<Ts>::type...> >
        when_any(
          BOOST_THREAD_FWD_REF(T) f,
          BOOST_THREAD_FWD_REF(Ts) ...futures);
#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATES
#endif // BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY

#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
  template <class>
  friend class packaged_task;
#else
  friend class packaged_task<T>;
#endif

  friend class boost::detail::future_waiter;

  template <typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
  boost::detail::make_future_async_shared_state(
    BOOST_THREAD_FWD_REF(C) c);

  template <typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
  boost::detail::make_future_deferred_shared_state(
    BOOST_THREAD_FWD_REF(C) c);

  typedef typename base_type::move_dest_type move_dest_type;

  BOOST_THREAD_FUTURE(future_ptr future) : base_type(future) {}

public:
  BOOST_THREAD_MOVABLE_ONLY(BOOST_THREAD_FUTURE)
  typedef boost::future_state::state state;
  typedef T value_type;

  BOOST_CONSTEXPR BOOST_THREAD_FUTURE() {}
  BOOST_THREAD_FUTURE(boost::exception_ptr const& ex) : base_type(ex) {}
  ~BOOST_THREAD_FUTURE() {}

  BOOST_THREAD_FUTURE(
    BOOST_THREAD_RV_REF(BOOST_THREAD_FUTURE) that) BOOST_NOEXCEPT :
      base_type(boost::move(
        static_cast<base_type&>(BOOST_THREAD_RV(that)))) {}

  BOOST_THREAD_FUTURE& operator=(
    BOOST_THREAD_RV_REF(BOOST_THREAD_FUTURE) that) BOOST_NOEXCEPT {

    this->base_type::operator=(
      boost::move(
        static_cast<base_type&>(BOOST_THREAD_FUTURE(that))));

    return this;
  }

  shared_future<T> share() {
    return shared_future<T>(boost::move(*this));
  }

  void swap(BOOST_THREAD_FUTURE& that) {
    static_cast<base_type*>(this)->swap(that);
  }

// private
  void set_async() {
    this->future_->set_async();
  }

// private
  void set_deferred() {
    this->future_->set_deferred();
  }

  void run_if_is_deferred() {
    return this->future_->run_if_is_deferred();
  }

  void run_if_is_deferred_or_ready() {
    return this->future_->run_if_is_deferred_or_ready();
  }

  move_dest_type get() {
    if (this->future_.get() == 0) {
      boost::throw_exception(boost::future_uninitialized());
    }

    boost::unique_lock<boost::mutex> lock(this->future_->mutex_);

    if (!this->future_->valid(lock)) {
      boost::throw_exception(boost::future_uninitialized());
    }

#ifdef BOOST_THREAD_PROVIDES_FUTURE_INVALID_AFTER_GET
    this->future_->invalidate(lock);
#endif // BOOST_THREAD_PROVIDES_FUTURE_INVALID_AFTER_GET

    return this->future_->get(lock);
  }

  move_dest_type get_or(BOOST_THREAD_RV_REF(T) v) {
    if (this->future_.get() == 0) {
      boost::throw_exception(boost::future_uninitialized());
    }

    boost::unique_lock<boost::mutex> lock(this->future_->mutex_);

    if (!this->future_->valid(lock)) {
      boost::throw_exception(boost::future_uninitialized());
    }

    this->future_->wait(lock, false);

#ifdef BOOST_THREAD_PROVIDES_FUTURE_INVALID_AFTER_GET
    this->future_->invalidate(lock);
#endif // BOOST_THREAD_PROVIDES_FUTURE_INVALID_AFTER_GET

    if (this->future_->has_value(lock)) {
      return this->future_->get(lock);
    } else {
      return boost::move(v);
    }
  }

  move_dest_type get_or(T const& v) {
    if (this->future_.get() == 0) {
      boost::throw_exception(boost::future_uninitialized());
    }

    boost::unique_lock<boost::mutex> lock(this->future_->mutex_);

    if (!this->future_->valid(lock)) {
      boost::throw_exception(boost::future_uninitialized());
    }

    this->future_->wait(lock, false);

#ifdef BOOST_THREAD_PROVIDES_FUTURE_INVALID_AFTER_GET
    this->future_->invalidate(lock);
#endif //BOOST_THREAD_PROVIDES_FUTURE_INVALID_AFTER_GET

    if (this->future_->has_value(lock)) {
      return this->future_->get(lock);
    } else {
      return v;
    }
  }

#ifdef BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
  template <typename F>
  inline BOOST_THREAD_FUTURE<
    typename boost::result_of<
      F(BOOST_THREAD_FUTURE)>::type> then(BOOST_THREAD_FWD_REF(F) f);

  template <typename F>
  inline BOOST_THREAD_FUTURE<
    typename boost::result_of<
      F(BOOST_THREAD_FUTURE)>::type> then(
        boost::launch policy,
        BOOST_THREAD_FWD_REF(F) f);

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
  template <typename E, typename F>
  inline BOOST_THREAD_FUTURE<
    typename  boost::result_of<
      F(BOOST_THREAD_FUTURE)>::type> then(
        E& e,
        BOOST_THREAD_FWD_REF(F) f);
#endif //BOOST_THREAD_PROVIDES_EXECUTORS
#endif // BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION

#ifdef BOOST_THREAD_PROVIDES_FUTURE_UNWRAP
  inline BOOST_THREAD_FUTURE<T2> unwrap();
#endif // BOOST_THREAD_PROVIDES_FUTURE_UNWRAP
}; // BOOST_THREAD_FUTURE

template <typename T>
class shared_future : public boost::detail::basic_future<T> {
  typedef boost::detail::basic_future<T> base_type;
  typedef typename base_type::future_ptr future_ptr;

  friend class boost::detail::future_waiter;
  friend class boost::promise<T>;

#ifdef BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
  template <typename, typename, typename>
  friend struct boost::detail::future_async_continuation_shared_state;

  template <typename, typename, typename>
  friend struct boost::detail::future_deferred_continuation_shared_state;

  template <typename F, typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
  boost::detail::make_future_async_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
  boost::detail::make_future_sync_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
  boost::detail::make_future_deferred_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c);
#endif // BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION

#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
  template <class>
  friend class packaged_task;
#else
  friend class packaged_task<T>;
#endif

  shared_future(future_ptr future) : base_type(future) {}

public:
  BOOST_THREAD_COPYABLE_AND_MOVABLE(shared_future)
  typedef T value_type;

  shared_future(shared_future const& that) : base_type(that.future_) {}

  typedef boost::future_state::state state;

  BOOST_CONSTEXPR shared_future() {}

  shared_future(boost::exceptional_ptr const& ex) : base_type(ex) {}
  ~shared_future() {}

  shared_future& operator=(BOOST_THREAD_COPY_ASSIGN_REF(shared_future) that) {
    this->future_ = that.future_;
    return *this;
  }

  shared_future(BOOST_THREAD_RV_REF(shared_future) that) BOOST_NOEXCEPT :
    base_type(boost::move(
      static_cast<base_type&>(
        BOOST_THREAD_RV(that)))) {}

  shared_future(BOOST_THREAD_RV_REF(
    BOOST_THREAD_FUTURE<T>) that) BOOST_NOEXCEPT :
      base_type(boost::move(
      static_cast<base_type&>(
        BOOST_THREAD_RV(that)))) {}

  shared_future& operator=(
    BOOST_THREAD_RV_REF(shared_future) that) BOOST_NOEXCEPT {
    base_type::operator=(
      boost::move(static_cast<base_type&>(BOOST_THREAD_RV(that))));
    return *this;
  }

  shared_future& operator=(
    BOOST_THREAD_RV_REF(BOOST_THREAD_FUTURE<T>) that) BOOST_NOEXCEPT {
    base_type::operator=(
      boost::move(static_cast<base_type&>(BOOST_THREAD_RV(that))));
    return *this;
  }

  void swap(shared_future& that) BOOST_NOEXCEPT {
    static_cast<base_type*>(this)->swap(that);
  }

  bool run_if_is_deferred() {
    return this->future_->run_if_is_deferred();
  }

  bool run_if_is_deferred_or_ready() {
    return this->future_->run_if_is_deferred_or_ready();
  }

  typedef typename boost::detail::shared_state<T> shared_state_type;

  typename shared_state_type::shared_future_get_result_type get() const {
    if (!this->future_) {
      boost::throw_exception(boost::future_uninitialized());
    }

    return this->future_->get_result_type();
  }

  template <typename T2>
  typename boost::disable_if<
    boost::is_void<T2>,
    typename shared_state_type::shared_future_get_result_type>::type
  get_or(BOOST_THREAD_RV_REF(T2) v)  const {
    if (!this->future_) {
      boost::throw_exception(boost::future_uninitialized());
    }

    this->future_->wait();

    if (this->future_->has_value()) {
      return this->future_->get_result_type();
    } else {
      boost::move(v);
    }
  }

#ifdef BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
  template <typename F>
  inline BOOST_THREAD_FUTURE<typename boost::result_of<
    F(shared_future)>::type> then(
      BOOST_THREAD_FWD_REF(F) f) const;

  template <typename F>
  inline BOOST_THREAD_FUTURE<typename boost::result_of<
    F(shared_future)>::type> then(
      boost::launch policy,
      BOOST_THREAD_FWD_REF(F) f) const;

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
  template <typename E, typename F>
  inline BOOST_THREAD_FUTURE<typename boost::result_of<
    F(shared_future)>::type> then(
      E& e,
      BOOST_THREAD_FWD_REF(F) f) const;
#endif // BOOST_THREAD_PROVIDES_EXECUTORS
#endif // BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
}; //shared_future

BOOST_THREAD_DCL_MOVABLE_BEG(T)
boost::shared_future<T>
BOOST_THREAD_DCL_MOVABLE_END

template <typename R>
class promise {
  typedef typename boost::detail::shared_state<R> shared_state;
  typedef boost::shared_ptr<shared_state> future_ptr;
  typedef typename shared_state::source_reference_type source_reference_type;
  typedef typename shared_state::rvalue_source_type rvalue_source_type;
  typedef typename shared_state::move_dest_type move_dest_type;
  typedef typename shared_state::shared_future_get_result_type
    shared_future_get_result_type;

  future_ptr future_;
  bool future_obtained_;

#ifdef BOOST_THREAD_PROVIDES_PROMISE_LAZY
  void lazy_init() {
    if (!boost::atomic_load(&future_)) {
      future_ptr blank;

      boost::atomic_compare_exchange(
        &future_, &blank, future_ptr(new shared_state));
    }
  }
#endif // BOOST_THREAD_PROVIDES_PROMISE_LAZY

public:
  BOOST_THREAD_MOVABLE_ONLY(promise)

#ifdef BOOST_THREAD_PROVIDES_FUTURE_CTOR_ALLOCATORS
  template <typename Allocator>
  promise(boost::allocator_arg_t, Allocator alloc) {
    typedef typename Allocator::template rebind<shared_state>::other Alloc;
    typedef boost::thread_detail::allocator_destructor<Alloc> Dtor;

    Alloc alloc_(alloc);
    future_ = future_ptr(
      new(alloc_.allocate(1)) shared_state(), Dtor(alloc_, 1));
    future_obtained_ = false;
  }
#endif // BOOST_THREAD_PROVIDES_FUTURE_CTOR_ALLOCATORS

  promise() :
#ifdef BOOST_THREAD_PROVIDES_PROMISE_LAZY
    future_(),
#else
    future_(new shared_state()),
#endif // BOOST_THREAD_PROVIDES_PROMISE_LAZY
    future_obtained_(false) {}

  ~promise() {
    if (future_) {
      boost::unique_lock<boost::mutex> lock(future_->mutex_);

      if (!future_->done_ && !future_->is_constructed_) {
        future_->mark_exceptional_finish_internal(
          boost::copy_exception(boost::broken_promise()), lock);
      }
    }
  }

  promise(BOOST_THREAD_RV_REF(promise) that) BOOST_NOEXCEPT :
    future_(BOOST_THREAD_RV(that).future_),
    future_obtained_(BOOST_THREAD_RV(that).future_obtained_) {
    BOOST_THREAD_RV(that).future_.reset();
    BOOST_THREAD_RV(that).future_obtained_ = false;
  }

  promise& operator=(BOOST_THREAD_RV_REF(promise) that) BOOST_NOEXCEPT {
    future_ = BOOST_THREAD_RV(that).future_;
    future_obtained_ = BOOST_THRED_RV(that).future_obtained_;
    BOOST_THREAD_RV(that).future_.reset();
    BOOST_THREAD_RV(that).future_obtained_ = false;
    return *this;
  }

  void swap(promise& that) {
    future_.swap(that.future_);
    std::swap(future_obtained_, that.future_obtained_);
  }

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
  void set_executor(executor_ptr_type e) {
    lazy_init();

    if (future_.get() == 0) {
      boost::throw_exception(boost::promise_moved());
    }

    boost::lock_guard<boost::mutex> lock(future_->mutex_);
    future_->set_executor_policy(e, lock);
  }
#endif // BOOST_THREAD_PROVIDES_EXECUTORS

  BOOST_THREAD_FUTURE<R> get_future() {
    lazy_init();

    if (future_.get() == 0) {
      boost::throw_exception(boost::promise_moved());
    }

    if (future_obtained_) {
      boost::throw_exception(boost::future_already_retrieved());
    }

    future_obtained_ = true;

    return BOOST_THREAD_FUTURE<R>(future_);
  }

#ifdef BOOST_NO_CXX11_RVALUE_REFERENCES
  template <typename T>
  typename boost::enable_if<
    boost::is_copy_constructible<T>::value &&
    boost::is_same<
      R,
      T>::value,
    void>::type set_value(T const& v) {
    lazyinit();

    boost::unique_lock<boost::mutex> lock(future_->mutex_);

    if (future_->done_) {
      boost::throw_exception(boost::promise_already_satisfied());
    }

    future_->mark_finished_with_result_internal(v, lock);
  }
#else
  void set_value(source_reference_type v) {
    lazy_init();

    boost::unique_lock<boost::mutex> lock(future_->mutex_);

    if (future_->done_) {
      boost::throw_exception(boost::promise_already_satisfied());
    }

    future_->mark_finished_with_result_internal(v, lock);
  }
#endif // BOOST_NO_CXX11_RVALUE_REFERENCES

  void set_value(rvalue_source_type v) {
    lazy_init();

    boost::unique_lock<boost::mutex> lock(future_->mutex_);

    if (future_->done_) {
      boost::throw_exception(boost::promise_already_satisfied());
    }

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    future_->mark_finished_with_result_internal(boost::move(v), lock);
#else
    future_->mark_finished_with_result_internal(
      static_cast<rvalue_source_type(v), lock);
#endif // BOOST_NO_CXX11_RVALUE_REFERENCES
  }

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
  template <typename... Args>
  void emplace(BOOST_THREAD_FWD_REF(Args) ...args) {
    lazy_init();

    boost::unique_lock<boost::mutex> lock(future_->mutex_);

    if (future_->done_) {
      boost::throw_exception(boost::promise_already_satisfied());
    }

    future_->mark_finished_with_result_internal(
      lock, boost::forward<Args>(args)...);
  }
#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATES

  void set_exception(boost::exception_ptr ex) {
    lazy_init();

    boost::unique_lock<boost::mutex> lock(future_->mutex_);

    if (future_->done_) {
      boost::throw_exception(boost::promise_already_satisfied());
    }

    future_->mark_exceptional_finish_internal(ex, lock);
  }

  template <typename Ex>
  void set_exception(Ex ex) {
    set_exception(boost::copy_exception(ex));
  }

#ifdef BOOST_NO_CXX11_RVALUE_REFERENCES
  template <typename T>
  typename boost::enable_if<
    boost::is_copy_constructible<T>::value &&
    boost::is_same<
      R,
      T>::value,
    void>::type set_value_at_thread_exit(T const& v) {
    if (future_->get() == 0) {
      boost::throw_exception(boost::promise_move());
    }

    future_->set_value_at_thread_exit(v);
  }
#else
  void set_value_at_thread_exit(source_reference_type v) {
    if (future_->get() == 0) {
      boost::throw_exception(boost::promise_moved());
    }

    future_->set_value_at_thread_exit();
  }
#endif // BOOST_NO_CXX11_RVALUE_REFERENCES

  void set_value_at_thread_exit(BOOST_THREAD_RV_REF(R) v) {
    if (future_->get() == 0) {
      boost::throw_exception(boost::promise_moved());
    }

    future_->set_value_at_thread_exit(boost::move(v));
  }

  void set_exception_at_thread_exit(boost::exception_ptr ex) {
    if (future_->get() == 0) {
      boost::throw_exception(boost::promise_moved());
    }

    future_->set_exception_at_thread_exit(ex);
  }

  template <typename Ex>
  void set_exception_at_thread_exit(Ex ex) {
    set_exception_at_thread_exit(boost::copy_exception(ex));
  }

  template <typename C>
  void set_wait_callback(C c) {
    lazy_init();
    future_->set_wait_callback(c, this);
  }

}; // promise

template <typename R>
class promise<R&> {
  typedef typename boost::detail::shared_state<R&> shared_state;
  typedef boost::shared_ptr<shared_state> future_ptr;

  future_ptr future_;
  bool future_obtained_;

#ifdef BOOST_THREAD_PROVIDES_PROMISE_LAZY
  void lazy_init() {
    if (!boost::atomic_load(&future_)) {
      future_ptr blank;

      boost::atomic_compare_exchange(
        &future_, &blank, future_ptr(new shared_state));
    }
  }
#endif // BOOST_THREAD_PROVIDES_PROMISE_LAZY

public:
  BOOST_THREAD_MOVABLE_ONLY(promise)

#ifdef BOOST_THREAD_PROVIDES_FUTURE_CTOR_ALLOCATORS
  template <typename Allocator>
  promise(boost::allocator_arg_t, Allocator alloc) {
    typedef typename Allocator::template rebind<shared_state>::other Alloc;
    typedef boost::thread_detail::allocator_destructor<Alloc> Dtor;

    Alloc alloc_(alloc);
    future_ = future_ptr(
      new(alloc_.allocator(1)) shared_state(), Dtor(alloc_, 1));
    future_obtained_ = false;
#endif // BOOST_THREAD_PROVIDES_FUTURE_CTOR_ALLOCATORS
  }

  promise() :
#ifdef BOOST_THREAD_PROVIDES_PROMISE_LAZY
    future_(),
#else
    future_(new shared_state()),
#endif // BOOST_THREAD_PROVIDES_PROMISE_LAZY
    future_obtained_(false) {}

  ~promise() {
    if (future_) {
      boost::unique_lock<boost::mutex> lock(future_->mutex_);

      if (!future_->done_ && !future_->is_constructed_) {
        future_->mark_exceptional_finish_internal(
          boost::copy_exception(boost::broken_promise()), lock);
      }
    }
  }

  promise(BOOST_THREAD_RV_REF(promise) that) BOOST_NOEXCEPT :
    future_(BOOST_THREAD_RV(that).future_),
    future_obtained_(BOOST_THREAD_RV(that).future_obtained_) {
    BOOST_THREAD_RV(that).future_.reset();
    BOOST_THREAD_RV(that).future_obtained_ = false;
  }

  promise& operator=(BOOST_THREAD_RV_REF(promise) that) BOOST_NOEXCEPT {
    future_ = BOOST_THREAD_RV(that).future_;
    future_obtained_ = BOOST_THREAD_RV(that).future_obtained_;
    BOOST_THREAD_RV(that).future_.reset();
    BOOST_THREAD_RV(that).future_obtained_ = false;
    return *this;
  }

  void swap(promise& that) {
    future_.swap(that.future_);
    std::swap(future_obtained_, that.future_obtained_);
  }

  BOOST_THREAD_FUTURE<R&> get_future() {
    lazy_init();

    if (future_.get() == 0) {
      boost::throw_exception(promise_moved());
    }

    if (future_obtained_) {
      boost::throw_exception(boost::future_already_retrieved());
    }

    future_obtained_ = true;

    return BOOST_THREAD_FUTURE<R&>(future_);
  }

  void set_value(R& v) {
    lazy_init();
    boost::unique_lock<boost::mutex> lock(future_->mutex);

    if (future_->done_) {
      boost::throw_exception(boost::promise_already_satisfied());
    }

    future_->mark_finished_with_result_internal(v, lock);
  }

  void set_exception(boost::exception_ptr ex) {
    lazy_init();
    boost::unique_lock<boost::mutex> lock(future_->mutex_);

    if (future_->done_) {
      boost::throw_exception(boost::promise_already_satisfied());
    }

    future_->mark_exceptional_finish_internal(ex, lock);
  }

  template <typename E>
  void set_exception(E ex) {
    set_exception(boost::copy_exception(ex));
  }

  void set_value_at_thread_exit(R& v) {
    if (future_.get() == 0) {
      boost::throw_exception(boost::promise_moved());
    }

    future_->set_value_at_thread_exit(v);
  }

  void set_exception_at_thread_exit(boost::exception_ptr ex) {
    if (future_->get() == 0) {
      boost::throw_exception(boost::promise_moved());
    }

    future_->set_exception_at_thread_exit(ex);
  }

  template <typename E>
  void set_exception_at_thread_exit(E ex) {
    set_exception_at_thread_exit(boost::copy_exception(ex));
  }

  template <typename C>
  void set_wait_callback(C c) {
    lazy_init();
    future_->set_wait_callback(c, this);
  }

}; // promise

template <>
class promise<void> {
  typedef typename boost::detail::shared_state<void> shared_state;
  typedef boost::shared_ptr<shared_state> future_ptr;

  future_ptr future_;
  bool future_obtained_;

#ifdef BOOST_THREAD_PROVIDES_PROMISE_LAZY
  void lazy_init() {
    if (!boost::atomic_load(&future_)) {
      future_ptr blank;

      boost::atomic_compare_exchange(
        &future_, &blank, future_ptr(new shared_state));
    }
  }
#endif // BOOST_THREAD_PROVIDES_PROMISE_LAZY

public:
  BOOST_THREAD_MOVABLE_ONLY(promise)

#ifdef BOOST_THREAD_PROVIDES_FUTURE_CTOR_ALLOCATORS
  template <typename Allocator>
  promise(boost::allocator_arg_t, Allocator alloc) {
    typedef typename Allocator::template rebind<shared_state>::other Alloc;
    typedef boost::thread_detail::allocator_destructor<Alloc> Dtor;

    Alloc alloc_(alloc);
    future_ = future_ptr(
      new(alloc_.allocate(1)) shared_state(), Dtor(alloc_, 1));
    future_obtained_ = false;
  }
#endif // BOOST_THREAD_PROVIDES_FUTURE_CTOR_ALLOCATORS

  promise() :
#ifdef BOOST_THREAD_PROVIDES_PROMISE_LAZY
    future_(),
#else
    future_(new shared_state()),
#endif // BOOST_THREAD_PROVIDES_PROMISE_LAZY
    future_obtained_(false) {}

  ~promise() {
    if (future_) {
      boost::unique_lock<boost::mutex> lock(future_->mutex_);

      if (!future_->done_ && !future_->is_constructed_) {
        future_->mark_exceptional_finish_internal(
          boost::copy_exception(boost::broken_promise()), lock);
      }
    }
  }

  promise(BOOST_THREAD_RV_REF(promise) that) BOOST_NOEXCEPT :
    future_(BOOST_THREAD_RV(that).future_),
    future_obtained_(BOOST_THREAD_RV(that).future_obtained_) {
    BOOST_THREAD_RV(that).future_.reset();
    BOOST_THREAD_RV(that).future_obtained_ = false;
  }

  promise& operator=(BOOST_THREAD_RV_REF(promise) that) BOOST_NOEXCEPT {
    future_ = BOOST_THREAD_RV(that).future_;
    future_obtained_ = BOOST_THREAD_RV(that).future_obtained_;
    BOOST_THREAD_RV(that).future_.reset();
    BOOST_THREAD_RV(that).future_obtained_ = false;
    return *this;
  }

  void swap(promise& that) {
    future_.swap(that.future_);
    std::swap(future_obtained_, that.future_obtained_);
  }

  BOOST_THREAD_FUTURE<void> get_future() {
    lazy_init();

    if (future_.get() == 0) {
      boost::throw_exception(boost::promise_moved());
    }

    if (future_obtained_) {
      boost::throw_exception(boost::future_already_retrieved());
    }

    return BOOST_THREAD_FUTURE<void>(future_);
  }

  void set_value() {
    lazy_init();

    boost::unique_lock<boost::mutex> lock(future_->mutex_);
    if (future_->done_) {
      boost::throw_exception(boost::promise_already_satisfied());
    }
    future_->mark_finished_with_result_internal(lock);
  }

  void set_exception(boost::exception_ptr p) {
    lazy_init();

    boost::unique_lock<boost::mutex> lock(future_->mutex_);
    if (future_->done_) {
      boost::throw_exception(boost::promise_already_satisfied());
    }
    future_->mark_exceptional_finish_internal(p, lock);
  }

  template <typename E>
  void set_exception(E e) {
    set_exception(boost::copy_exception(e));
  }

  void set_value_at_thread_exit() {
    if (future_.get() == 0) {
      boost::throw_exception(boost::promise_moved());
    }
    future_->set_value_at_thread_exit();
  }

  void set_exception_at_thread_exit(boost::exception_ptr e) {
    if (future_.get() == 0) {
      boost::throw_exception(boost::promise_moved());
    }
    future_->set_exception_at_thread_exit(e);
  }

  template <typename E>
  void set_exception_at_thread_exit(E e) {
    set_exception_at_thread_exit(boost::copy_exception(e));
  }

  template <typename C>
  void set_wait_callback(C c) {
    lazy_init();
    future_->set_wait_callback(c, this);
  }
}; // promise
} // boost

#ifdef BOOST_THREAD_PROVIDES_FUTURE_CTOR_ALLOCATORS
namespace boost {
namespace container {
  template <typename R, typename Allocator>
  struct uses_allocator<boost::promise<R>, Allocator> : true_type {};
} // container
} // boost
#ifndef BOOST_NO_CXX11_ALLOCATOR
namespace std {
  template <typename R, typename Allocator>
  struct uses_allocator<boost::promise<R>, Allocator> : true_type {};
} // std
#endif // BOOST_NO_CXX11_ALLOCATOR
#endif // BOOST_THREAD_PROVIDES_FUTURE_CTOR_ALLOCATORS

namespace boost {

BOOST_THREAD_DCL_MOVABLE_BEG(T)
boost::promise<T>
BOOST_THREAD_DCL_MOVABLE_END

namespace detail {
#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
template <typename R>
struct task_base_shared_state;
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
template <typename R, typename... Ts>
struct task_base_shared_state<R(Ts...)> :
#else
template <typename R>
struct task_base_shared_state<R()> :
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
#else
template <typename R>
struct task_base_shared_state :
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK

boost::detail::shared_state<R> {
  bool started_;

  task_base_shared_state() : started_(false) {}

  void reset() {
    started_ = false;
    this->validate();
  }

#if defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK) &&                 \
    defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
  virtual void do_run(BOOST_THREAD_RV_REF(Ts) ...ts) = 0;
  void run(BOOST_THREAD_RV_REF(Ts) ...ts) {
#else
  virtual void do_run() = 0;
  void run() {
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
       // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
    {
      boost::lock_guard<boost::mutex> lock(this->mutex_);
      if (started_) {
        boost::throw_exception(boost::task_already_started());
      }

      started_ = false;
    }
#if defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK) &&                 \
    defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
    do_run(boost::move(ts)...);
#else
    do_run();
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
       // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
  } // run
#if defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK) &&                 \
    defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
  virtual void do_apply(BOOST_THREAD_RV_REF(Ts) ...ts) = 0;
  void apply(BOOST_THREAD_RV_REF(Ts) ...ts) {
#else
  virtual void do_apply() = 0;
  void apply() {
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
       // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
    {
      boost::lock_guard<boost::mutex> lock(this->mutex_);
      if (started_) {
        boost::throw_exception(boost::task_already_started());
      }

      started_ = true;
    }
#if  defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK) &&                \
     defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
    do_apply(boost::move(ts)...);
#else
    do_apply();
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
       // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
  } // apply

  void owner_destroyed() {
    boost::unique_lock<boost::mutex> lock(this->mutex_);
    if (!started_) {
      started_ = true;
      this->mark_exceptional_finish_internal(
        boost::copy_exception(boost::broken_promise()), lock);
    }
  }
}; // task_base_shared_state

#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
template <typename F, typename R>
struct task_shared_state;
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
template <typename F, typename R, typename ...Ts>
struct task_shared_state<F, R(Ts...)> :
  task_base_shared_state<R(Ts...)> {
#else
template <typename F, typename R>
struct task_shared_state<F, R()> :
  task_base_shared_state<R()> {
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
#else
template <typename F, typename R>
struct task_shared_state :
  task_base_shared_state<R> {
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
private:
  task_shared_state(task_shared_state&);
public:
  F f_;
  task_shared_state(F const& f) : f_(f) {}
  task_shared_state(BOOST_THREAD_RV_REF(F) f) : f_(boost::move(f)) {}

  F callable() {
    return boost::move(f_);
  }

#if defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK) &&                 \
    defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
  void do_apply(BOOST_THREAD_RV_REF(Ts) ...ts) {
    try {
      this->set_value_at_thread_exit(f_(boost::move(ts)...));
#else
  void do_apply() {
    try {
      this->set_value_at_thread_exit(f_());

#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
       // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
    } catch (...) {
      this->set_exception_at_thread_exit(boost::current_exception());
    }
  } // do_apply

#if defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK) &&                 \
    defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
  void do_run(BOOST_THREAD_RV_REF(Ts) ...ts) {
    try {
      this->mark_finished_with_result(f_(boost::move(ts)...));
#else
  void do_run() {
    try {
#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      R r(f_());
      this->mark_finished_with_result(boost::move(r));
#else
      this->mark_finished_with_result(f_());
#endif // BOOST_NO_CXX11_RVALUE_REFERENCES
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
       // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
    } catch (...) {
      this->mark_exceptional_finish();
    }
  }
}; // task_shared_state

#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
template <typename F, typename R, typename... Ts>
struct task_shared_state<F, R&(Ts...)> :
  task_base_shared_state<R&(Ts...)>
#else
template <typename F, typename R>
struct task_shared_state<F, R&()> :
  task_base_shared_state<R&>
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
#else
template <typename T, typename R>
struct task_shared_state<F, R&> :
  task_base_shared_state<R&>
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
{
private:
  task_shared_state(task_shared_state&);
public:
  F f_;
  task_shared_state(F const& f) : f_(f) {}
  task_shared_state(BOOST_THREAD_RV_REF(F) f) : f_(boost::move(f)) {}

  F callable() {
    return f_;
  }

#if defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK) &&                 \
    defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
  void do_apply(BOOST_THREAD_RV_REF(Ts) ...ts) {
    try {
      this->set_value_at_thread_exit(f_(boost::move(ts)...));
#else
  void do_apply() {
    try {
      this->set_value_at_thread_exit(f_());
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
       // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
    } catch (...) {
      this->set_exceptional_at_thread_exit(boost::current_exception());
    }
  } // do_apply

#if defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK) &&                 \
    defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
  void do_run(BOOST_THREAD_RV_REF(Ts) ...ts) {
    try {
      this->mark_finished_with_result(f_(boost::move(ts)...));
#else
  void do_run() {
    try {
      R& r(f_());
      this->mark_finished_with_result(r);
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
       // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
    } catch (...) {
      this->mark_exceptional_finish();
    }
  } // do_run
}; // task_shared_state

#ifdef BOOST_THREAD_RVALUE_REFERENCES_DONT_MATCH_FUNCTION_PTR
#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
template <typename R, typename... Ts>
struct task_shared_state<R(*)(Ts...), R(Ts...)> :
  task_base_shared_state<R(Ts...)>
#else
template <typename R>
struct task_shared_state<R(*)(), R()> :
  task_base_shared_state<R()>
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
#else
template <typename R>
struct task_shared_state<R(*)(), R> :
  task_base_shared_state<R>
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
{
private:
  task_shared_state(task_shared_state&);
#if defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK) &&                 \
    defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
  typedef R (*CallableType)(Ts ...);
#else
  typedef R (*Callabletype)();
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
       // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
public:
  CallableType f_;
  task_shared_state(CallableType f) : f_(f) {}

  CallableType callable() {
    return f_;
  }

#if defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK) &&                 \
    defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
  void do_apply(BOOST_THREAD_RV_REF(Ts) ...ts) {
    try {
      this->set_value_at_thread_exit(f_(boost::move(ts)...));
#else
  void do_apply() {
    try {
      R r(f_());
      this->set_value_at_thread_exit(boost::move(r));
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
       // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
    } catch (...) {
      this->set_exception_at_thread_exit(boost::current_exception());
    }
  } // do_apply

#if defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK) &&                 \
    defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
  void do_run(BOOST_THREAD_RV_REF(Ts) ...ts) {
    try {
      this->mark_finished_with_result(f_(boost::move(ts)...));
#else
  void do_run() {
    try {
      R r(f_());
      this->mark_finished_with_result(boost::move(r));
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
       // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
    } catch (...) {
      this->mark_exceptional_finish();
    }
  }
}; // task_shared_state

#if defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK)
#if defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
template <typename R, typename... Ts>
struct task_shared_state<R&(*)(Ts...), R&(Ts...)> :
  task_base_shared_state<R&(Ts...)>
#else
template <typename R>
struct task_shared_state<R&(*)(), R&()> :
  task_base_shared_state<R&()>
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
#else
template <typename R>
struct task_shared_state<R&(*)(), R&> :
  task_base_shared_state<R&>
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
{
private:
  task_shared_state(task_shared_state&);
public:
#if defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK) &&                 \
    defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
  typedef R& (*CallableType)(BOOST_THREAD_RV_REF(Ts)...);
#else
  typedef R& (*CallableType)();
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
       // BOOST_THREAD_PROVIDES_VARIADIC_THREAD

public:
#if defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK) &&                 \
    defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
  void do_apply(BOOST_THREAD_RV_REF(Ts) ...ts) {
    try {
      this->set_value_at_thread_exit(f_(boost::move(ts)...));
#else
  void do_apply() {
    try {
      this->set_value_at_thread_exit(f_());
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
       // BOOST_THREAD_PROVIDES_VAIRADIC_THREAD
    } catch (...) {
      this->set_exception_at_thread_exit(boost::current_exception());
    }
  } // do_apply

#if defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK) &&                 \
    defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
  void do_run(BOOST_THREAD_RV_REF(Ts) ...ts) {
    try {
      this->mark_finished_with_result(f_(boost::move(ts)...));
#else
  void do_run() {
    try {
      this->mark_finished_with_result(f_());
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
       // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
    } catch (...) {
      this->mark_exceptional_finish();
    }
  } // do_run
}; // task_shared_state
#endif // BOOST_THREAD_RVALUE_REFERENCES_DONT_MATCH_FUNCTION_PTR

#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
template <typename F, typename... Ts>
struct task_shared_state<F, void(Ts...)> :
  task_base_shared_state<void(Ts...)>
#else
template <typename F>
struct task_shared_state<F, void()> :
  task_base_shared_state<void()>
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
#else
template <typename F>
struct task_shared_state<F, void> :
  task_base_shared_state<void>
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
{
private:
  task_shared_state(task_shared_state&);
public:
  typedef F CallableType;
  F f_;
  task_shared_state(F const& f) : f_(f) {}
  task_shared_state(BOOST_THREAD_RV_REF(F) f) : f_(boost::move(f)) {}

  F callable() {
    return boost::move(f_);
  }

#if defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK) &&                 \
    defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
  void do_apply(BOOST_THREAD_RV_REF(Ts) ...ts) {
    try {
      f_(boost::move(ts)...);
#else
  void do_apply() {
    try {
      f_();
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
       // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
      this->set_value_at_thread_exit();
    } catch (...) {
      this->set_exception_at_thread_exit(boost::current_exception());
    }
  } // do_apply

#if defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK) &&                 \
    defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
  void do_run(BOOST_THREAD_RV_REF(Ts) ...ts) {
    try {
      f_(boost::move(ts)...);
#else
  void do_run() {
    try {
      f_();
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
       // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
      this->mark_finished_with_internal();
    } catch (...) {
      this->mark_exceptional_finish();
    }
  }
}; // task_shared_state

#if defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK)
#if defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
template <typename... Ts>
struct task_shared_state<void(*)(Ts...), void(Ts...)> :
  task_base_shared_state<void(Ts...)>
#else
template <>
struct task_shared_state<void(*)(), void()> :
  task_base_shared_state<void()>
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
#else
template <>
struct task_shared_state<void(*)(), void> :
  task_base_shared_state<void>
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
{
private:
  task_shared_state(task_shared_state&);
#if defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED) &&                      \
    defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
  typedef void (*CallableType)(Ts...);
#else
  typedef void (*CallableType)();
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED
       // BOOST_THREAD_PROVIDES_VARIADIC_THREAD

public:
  CallableType f_;
  task_shared_state(CallableType f) : f_(f) {}

  CallableType callable() {
    return f_;
  }

#if defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK) &&                 \
    defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
  void do_apply(BOOST_THREAD_RV_REF(Ts) ...ts) {
    try {
      f_(boost::move(ts)...);
#else
  void do_apply() {
    try {
      f_();
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
       // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
      this->set_value_at_thread_exit();
    } catch (...) {
      this->set_exception_at_thread_exit(boost::current_exception());
    }
  } // do_apply

#if defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK) &&                 \
    defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
  void do_run(BOOST_THREAD_RV_REF(Ts) ...ts) {
    try {
      f_(boost::move(ts)...);
#else
  void do_run() {
    try {
      f_();
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
       // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
      this->mark_finished_with_result();
    } catch (...) {
      this->mark_exceptional_finish();
    }
  } // do_run
};
} // detail

#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
template <typename R, typename... Ts>
class packaged_task<R(Ts...)> {
  typedef boost::shared_ptr<
    boost::detail::task_base_shared_state<R(Ts...)> > task_ptr;
  boost::shared_ptr<
    boost::detail::task_base_shared_state<R(Ts...)> > task_;
#else
template <typename R>
class packaged_task<R()> {
  typedef boost::shared_ptr<
    boost::detail::task_base_shared_state<R()> > task_ptr;
  boost::shared_ptr<
    boost::detail::task_base_shared_state<R()> > task_;
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
#else
template <typename R>
class packaged_task {
  typedef boost::shared_ptr<
    boost::detail::task_base_shared_state<R> > task_ptr;
  boost::shared_ptr<
    boost::detail::task_base_shared_state<R> > task_;
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK

  bool future_obtained_;
  struct dummy;

public:
  typedef R result_type;
  BOOST_THREAD_MOVABLE_ONLY(packaged_task)

  packaged_task() : future_obtained_(false) {}

#ifdef BOOST_THREAD_RVALUE_REFERENCES_DONT_MATCH_FUNCTION_PTR
#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
explicit packaged_task(
  R(*f)(),
  BOOST_THREAD_FWD_REF(Ts)... ts) {
  typedef R(*FR)(BOOST_THREAD_FWD_REF(Ts)...);
  typedef boost::detail::task_shared_state<
    FR, R(Ts...)> task_shared_state_type;

  task_ = task_ptr(
    new task_shared_state_type(f, boost::move(ts)...));
  future_obtained_ = false;
}
#else
explicit packaged_task(R(*f)()) {
  typedef R(*FR)();
  typedef boost::detail::task_shared_state<FR, R()> task_shared_state_type;

  task_ = task_ptr(new task_shared_state_type(f));
  future_obtained_ = false;
}
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
#else
explicit packaged_task(R(*f)() {
  typedef R(*FR)();
  typedef boost::detail::task_shared_state<FR, R> task_shared_state_type;

  task_ = task_ptr(new task_shared_state_type(f));
  future_obtained_ = false;
}
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
#endif // BOOST_THREAD_RVALUE_REFERENCES_DONT_MATCH_FUNCTION_PTR

#ifdef BOOST_NO_CXX11_RVALUE_REFERENCES
template <typename F>
explicit packaged_task(
  BOOST_THREAD_FWD_REF(F) F,
  typename boost::disable_if<
    boost::is_same<
      typename boost::decay<F>::type,
      packaged_task>,
  dummy* >::type = 0) {
  typedef typename boost::decay<F>::type FR;

#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
  typedef boost::detail::task_shared_state<
    FR, R(Ts...)> task_shared_state_type;
#else
  typedef boost::detail::task_shared_state<FR, R()> task_shared_state_type;
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
#else
  typedef boost::detail::task_shared_state<FR, R> task_shared_state_type;
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK

  task_ = task_ptr(
    new task_shared_state_type(boost::forward<F>(f)));
  future_obtained_ = false;
}
#else
template <typename F>
explicit packaged_task(F const& f,
  typename boost::disable_if<
    boost::is_same<
      typename boost::decay<F>::type,
      packaged_task>,
  dummy* >::type = 0) {
#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
  typedef boost::detail::task_shared_state<F, R(Ts...)> task_shared_state_type;
#else
  typedef boost::detail::task_shared_state<F, R()> task_shared_state_type;
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
#else
  typedef boost::detail::task_shared_state<F, R> task_shared_state_type;
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK

  task_ = task_ptr(new task_shared_state_type(f));
  future_obtained_ = false;
}

template <typename F>
explicit packaged_task(BOOST_THREAD_RV_REF(F) f) {
#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
  typedef boost::detail::task_shared_state<F, R(Ts...)> task_shared_state_type;
  task_ = task_ptr(new task_shared_state_type(boost::move(f)));
#else
  typedef boost::detail::task_shared_state<F, R()> task_shared_state_type;
  task_ = task_ptr(new task_shared_state_type(boost::move(f)));
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
#else
  typedef boost::detail::task_shared_state<F, R> task_shared_state_type;
  task_ = task_ptr(new task_shared_state_type(boost::move(f)));
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
  future_obtained_ = false;
}
#endif // BOOST_NO_CXX11_RVALUE_REFERENCES

#ifdef BOOST_THREAD_PROVIDES_FUTURE_CTOR_ALLOCATORSS
#ifdef BOOST_THREAD_RVALUE_REFERENCES_DONT_MATCH_FUTURE_PTR
template <typename Allocator>
packaged_task(boost::allocator_arg_t, Allocator alloc, R(*f)()) {
  typedef R(*FR)();
#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
  typedef boost::detail::task_shared_state<
    FR, R(Ts...)> task_shared_state_type;
#else
  typedef boost::detail::task_shared_state<FR, R()> task_shared_state_type;
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
#else
  typedef boost::detail::task_shared_state<FR, R> task_shared_state_type;
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK

  typedef typename Allocator::template rebind<
    task_shared_state_type>::other Alloc;
  typedef boost::thread_detail::allocator_destructor<Alloc> Dtor;

  Alloc alloc_(alloc);
  task_ = task_ptr(
    new(alloc_.allocate(1)) task_shared_state_type(f), Dtor(alloc_, 1));
  future_obtained_ = false;

#endif // BOOST_THREAD_RVALUE_REFERENCES_DONT_MATCH_FUTURE_PTR
#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
template <typename F, typename Allocator>
packaged_task(
  boost::allocator_arg_t,
  Allocator alloc,
  BOOST_THREAD_FWD_REF(F) f) {
  typedef typename boost::decay<F>::type FR;

#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
  typedef boost::detail::task_shared_state<
    FR, R(Ts...)> task_shared_state_type;
#else
  typedef boost::detail::task_shared_state<FR, R()> task_shared_state_type;
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
#else
  typedef boost::detail::task_shared_state<FR, R> task_shared_state_type;
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK

  typedef typename Allocator::template rebind<
    task_shared_state_type>::other Alloc;
  typedef boost::thread_detail::allocator_destructor<Alloc> Dtor;

  Alloc alloc_(alloc);
  task_ = task_ptr(
    new(alloc_.allocate(1)) task_shared_state_type(
      boost::forward<F>(f)), Dtor(alloc_, 1));
  future_obtained_ = false;
}
#else
template <typename F, typename Allocator>
packaged_task(boost::allocator_arg_t, Allocator alloc, const F& f) {
#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
  typedef boost::detail::task_shared_state<F, R(Ts...)> task_shared_state_type;
#else
  typedef boost::detail::task_shared_state<F, R()> task_shared_state_type;
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
#else
  typedef boost::detail::task_shared_state<F, R> task_shared_state_type;
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK

  typedef typename Allocator::template rebind<
    task_shared_state_type>::other Alloc;
  typedef boost::thread_detail::allocator_destructor<Alloc> Dtor;

  Alloc alloc_(alloc);
  task_ = task_ptr(
    new(alloc_.allocate(1)) task_shared_state_type(
      boost::move(f)), Dtor(alloc_, 1));
  future_obtained_ = false;
}
#endif // BOOST_NO_CXX11_RVALUE_REFERENCES
#endif // BOOST_THREAD_PROVIDES_FUTURE_CTOR_ALLOCATORS

~packaged_task() {
  if (task_) {
    task_->owner_destroyed();
  }
}

packaged_task(BOOST_THREAD_RV_REF(packaged_task) that) BOOST_NOEXCEPT :
  future_obtained_(BOOST_THREAD_RV(that).future_obtained_) {
  task_->swap(BOOST_THREAD_RV(that).task_);
  BOOST_THREAD_RV(that).future_obtained_ = false;
}

packaged_task& operator=(
  BOOST_THREAD_RV_REF(packaged_task) that) BOOST_NOEXCEPT {
#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
  packaged_task temp(boost::move(that));
#else
  packaged_task temp(
    static_cast<BOOST_THREAD_RV_REF(packaged_task)>(that));
#endif // BOOST_NO_CXX11_RVALUE_REFERENCES
  swap(temp);
  return *this;
}

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
void set_executor(executor_ptr_type ex) {
  if (!valid()) {
    boost::throw_exception(boost::task_moved());
  }

  boost::lock_guard<boost::mutex> lock(task_->mutex_);
  task_->set_executor_policy(ex, lock);
}
#endif // BOOST_THREAD_PROVIDES_EXECUTORS

void reset() {
  if (!valid()) {
    boost::throw_exception(
      boost::future_error(
        boost::system::make_error_code(
          boost::future_errc::no_state)));
  }

  task_->reset();
  future_obtained_ = false;
}

void swap(packaged_task that) BOOST_NOEXCEPT {
  task_.swap(that.task_);
  std::swap(future_obtained_, that.future_obtained_);
}

bool valid() const BOOST_NOEXCEPT {
  return task_.get() != 0;
}

BOOST_THREAD_FUTURE<R> get_future() {
  if (!task_) {
    boost::throw_exception(boost::task_moved());
  } else if (!future_obtained_) {
    future_obtained_ = true;
    return BOOST_THREAD_FUTURE<R>(task_);
  } else {
    boost::throw_exception(boost::future_already_retrieved());
  }
}

#if defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK) &&                 \
    defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
void operator()(Ts ...ts) {
  if (!task_) {
    boost::throw_exception(boost::task_moved());
  }

  task_->run(boost::move(ts)...);
}

void make_read_at_thread_exit(Ts... ts) {
  if (!task_) {
    boost::throw_exception(boost::task_moved());
  }

  if (task_->has_value()) {
    boost::throw_exception(boost::promise_already_satisfied());
  }

  task_->apply(boost::move(ts)...);
}
#else
void operator()() {
  if (!task_) {
    boost::trow_exception(boost::task_moved());
  }

  task_->run();
}

void make_ready_at_thread_exit() {
  if (!task_) {
    boost::throw_exception(boost::task_moved());
  }

  if (task_->has_value()) {
    boost::throw_exception(boost::promise_already_satisfied());
  }

  task_->apply();
}
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
       // BOOST_THREAD_PROVIDES_VARIADIC_THREAD

template <typename F>
void set_wait_callback(F f) {
  task_->set_wait_callback(f, this);
}
}; // packaged_task
} // boost

#if defined BOOST_THREAD_PROVIDES_FUTURE_CTOR_ALLOCATORS
namespace boost {
namespace container {
template <typename R, typename Allocator>
struct uses_allocator<boost::packaged_task<R>, Allocator> : true_type {};
} // container
} // boost
#ifndef BOOST_NO_CXX11_ALLOCATOR
namespace std {
template <typename R, typename Allocator>
struct uses_allocator<boost::packaged_task<R>, Allocator> : true_type {};
} // std
#endif // BOOST_NO_CXX11_ALLOCATOR
#endif // BOOST_THREAD_PROVIDES_FUTURE_CTOR_ALLOCATORS

namespace boost {

BOOST_THREAD_DCL_MOVABLE_BEG(T)
boost::packaged_task<T>
BOOST_THREAD_DCL_MOVABLE_END

namespace detail {

/* make_future_async_shared_state */
template <typename R, typename F>
BOOST_THREAD_FUTURE<R>
  make_future_async_shared_state(BOOST_THREAD_FWD_REF(F) f) {
  boost::shared_ptr<
    boost::detail::future_async_shared_state<R, F> > h(
      new boost::detail::future_async_shared_state<R, F>());
  h->init(boost::forward<F>(f));

  return BOOST_THREAD_FUTURE<R>(h);
}

/* make_future_deferred_shared_state */
template <typename R, typename F>
BOOST_THREAD_FUTURE<R>
  make_future_deferred_shared_state(BOOST_THREAD_FWD_REF(F) f) {
  boost::shared_ptr<
    boost::detail::future_deferred_shared_state<R, F> >
      h(new boost::detail::future_async_shared_state<R, F>(
        boost::forward<F>(f)));

  return  BOOST_THREAD_FUTURE<R>(h);
}
} // detail

#ifdef BOOST_THREAD_RVALUE_REFERENCES_DONT_MATCH_FUNCTION_PTR
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
template <typename R, typename... Ts>
BOOST_THREAD_FUTURE<R>
  async(
    boost::launch policy,
    R(*f)(BOOST_THREAD_FWD_REF(Ts)...),
    BOOST_THREAD_FWD_REF(Ts)... ts) {
    typedef R(*F)(BOOST_THREAD_FWD_REF(Ts)...);
    typedef boost::detail::invoker<
      typename boost::decay<F>::type,
      typename boost::decay<Ts>::type...> callback_type;
    typedef typename callback_type::result_type result_type;

    if (boost::underlying_cast<int>(policy) &&
        int(boost::launch::async)) {
      return BOOST_THREAD_MAKE_RV_REF(
        boost::detail::make_future_async_shared_state<result_type>(
          callback_type(
            f,
            boost::thread_detail::decay_copy(
              boost::forward<Ts>(ts))...)));
    } else if (boost::underlying_cast<int>(policy) &&
               int(boost::launch::async)) {
      return BOOST_THREAD_MAKE_RV_REF(
        boost::detail::make_future_deferred_shared_state<result_type>(
          callback_type(
            f,
            boost::thread_detail::decay_copy(
              boost::forward<Ts>(ts))...)));
    } else {
      std::terminate();
  }
}
#else
template <typename R>
BOOST_THREAD_FUTURE<R>
  async(boost::launch policy, R(*f)()) {
#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
  typedef boost::packaged_task<R()> packaged_task_type;
#else
  typedef boost::packaged_task<R> packaged_task_type;
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK

  if (boost::underlying_cast<int>(policy) &&
      int(boost::launch::async)) {
    packaged_task_type task_type(f);
    BOOST_THREAD_FUTURE<R> r =
      BOOST_THREAD_MAKE_RV_REF(task_type.get_future());
    r.set_async();
    boost::thread(boost::move(task_type)).detach();
    return boost::move(r);
  } else if (boost::underlying_cast<int>(policy) &&
             int(boost::launch::deferred) {
    std::terminate();
  } else {
    std::terminate();
  }
}
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
#endif // BOOST_THREAD_RVALUE_REFERENCES_DONT_MATCH_FUNCTION_PTR

#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
template <typename F, typename... Ts>
BOOST_THREAD_FUTURE<typename boost::result_of<
  typename boost::decay<F>::type(
    typename boost::decay<Ts>::type...)>::type>
  async(
    boost::launch policy,
    BOOST_THREAD_FWD_REF(F) f,
    BOOST_THREAD_FWD_REF(Ts) ...ts) {
    typedef boost::detail::invoker<
      typename boost::decay<F>::type,
      typename boost::decay<Ts>::type...> callback_type;
    typedef typename callback_type::result_type result_type;

  if (boost::underlying_cast<int>(policy) &&
      int(boost::launch::async)) {
    return BOOST_THREAD_MAKE_RV_REF(
      boost::detail::make_future_async_shared_state<result_type>(
        callback_type(
          boost::thread_detail::decay_copy(boost::forward<F>(f)),
          boost::thread_detail::decay_copy(boost::forward<Ts>(ts))...)));
  } else if (boost::underlying_cast<int>(policy) &&
             int(boost::launch::deferred)) {
    return BOOST_THREAD_MAKE_RV_REF(
      boost::detail::make_future_deferred_shared_state<result_type>(
        callback_type(
          boost::thread_detail::decay_copy(boost::forward<F>(f)),
          boost::thread_detail::decay_copy(boost::forward<Ts>(ts))...)));
  } else {
    std::terminate();
  }
}
#else
template <typename F>
BOOST_THREAD_FUTURE<
  typename boost::result_of<
    typename boost::decay<F>::type()>::type
  async(
    boost::launch policy,
    BOOST_THREAD_FWD_REF(F) f) {

    typedef typename boost::result_of<
      typename boost::decay<F>::type()>::type R;
#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
    typedef boost::packaged_task<R()> packaged_task_type;
#else
    typedef boost::packaged_task<R> packaged_task_type;
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK

  if (boost::underlying_cast<int>(policy) &&
      int(boost::launch::async)) {
    packaged_task_type task_type(boost::forward<F>(f));
    BOOST_THREAD_FUTURE<R> r = task_type.get_future();
    r.set_async();
    boost::thread(boost::move(task_type)).detach;
    return boost::move(r);
  } else if (boost::underlying_cast<int>(policy) &&
             int(boost::launch::deferred)) {
    std::terminal();
  } else {
    std::terminal();
  }
}
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
namespace detail {

/* shared_state_nullary_task */
template <typename R, typename F>
struct shared_state_nullary_task {
  typedef boost::shared_ptr<boost::detail::shared_state_base> storage_type;
  storage_type st_;
  F f_;

  shared_state_nullary_task(storage_type st, BOOST_THREAD_FWD_REF(F) f) :
    st_(st), f_(f) {}

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
  // copy
  BOOST_THREAD_COPYABLE_AND_MOVABLE(shared_state_nullary_task)
  shared_state_nullary_task(shared_state_nullary_task const& s) :
    st_(s.st_), f_(s.f_) {}

  shared_state_nullary_task& operator=(BOOST_THREAD_COPY_ASSIGN_REF(
    shared_state_nullary_task) s) {

    if (this != &s) {
      st_ = s.st_;
      f_ = s.f_;
    }
    return *this;
  }

  // move
  shared_state_nullary_task(BOOST_THREAD_RV_REF(shared_state_nullary_task) s) :
    st_(s.st_), f_(boost::move(s.f_)) {
    s.st_.reset();
  }

  shared_state_nullary_task& operator=(BOOST_THREAD_RV_REF(
    shared_state_nullary_task) s) {

    if (this != &s) {
      st_ = s.st_;
      f_ = boost::move(s.f_);
      s.st_.reset();
    }
    return *this;
  }
#endif // BOOST_NO_CXX11_RVALUE_REFERENCES

  void operator()() {

    boost::shared_ptr<boost::detail::shared_state<R> > st =
      static_pointer_cast<boost::detail::shared_state<R> >(st_);

    try {
      st->mark_finished_with_result(f_());
    } catch (...) {
      st->mark_exceptional_finish();
    }
  }
};

template <typename F>
struct shared_state_nullary_task<void, F> {
  typedef boost::shared_ptr<boost::detail::shared_state_base> storage_type;
  storage_type st_;
  F f_;

  shared_state_nullary_task(storage_type st, BOOST_THREAD_FWD_REF(F) f) :
    st_(st), f_(f) {}

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
  // copy
  BOOST_THREAD_COPYABLE_AND_MOVABLE(shared_state_nullary_task)
  shared_state_nullary_task(shared_state_nullary_task const& s) :
    st_(s.st_), f_(s.f_) {}

  shared_state_nullary_task& operator=(BOOST_THREAD_COPY_ASSIGN_REF(
    shared_state_nullary_task) s) {

    if (this != s) {
      st_ = s.st_;
      f_ = s.f_;
    }
    return *this;
  }

  // move
  shared_state_nullary_task(
    BOOST_THREAD_RV_REF(shared_state_nullary_task) s) BOOST_NOEXCEPT :
    st_(s.st_), f_(boost::move(s.f_)) {
    s.st_.reset();
  }

  shared_state_nullary_task& operator=(BOOST_THREAD_RV_REF(
    shared_state_nullary_task) s) BOOST_NOEXCEPT {

    if (this != s) {
      st_ = s.st_;
      f_ = boost::move(s.f_);
      s.st_.reset();
    }
    return *this;
  }
#endif // BOOST_NO_CXX11_RVALUE_REFERENCES
  void operator()() {

    boost::shared_ptr<boost::detail::shared_state<void> > st =
      static_pointer_cast<boost::detail::shared_state<void> >(st_);

    try {
      f_();
      st->mark_finished_with_result();
    } catch (...) {
      st->mark_exceptional_finish();
    }
  }
};
} // detail

BOOST_THREAD_DCL_MOVABLE_BEG2(R, F)
boost::detail::shared_state_nullary_task<R, F>
BOOST_THREAD_DCL_MOVABLE_END

namespace detail {

/* future_executor_shared_state */
template <typename R>
struct future_executor_shared_state :
  boost::detail::shared_state<R> {
  typedef boost::detail::shared_state<R> base_type;

  future_executor_shared_state() {}

  template <typename F, typename Ex>
  void init(Ex& ex, BOOST_THREAD_FWD_REF(F) f) {

    typedef typename boost::decay<F>::type callback_type;

    this->set_executor_policy(
      boost::executor_ptr_type(new executor_ref<Ex>(ex)));

    boost::detail::shared_state_nullary_task<R, callback_type> t(
      this->shared_from_this(), boost::forward<F>(f));

    ex.submit(boost::move(t));
  }

  ~future_executor_shared_state() {}
};

/* make_future_executor_shared_state */
template <typename R, typename F, typename Ex>
BOOST_THREAD_FUTURE<R>
  make_future_executor_shared_state(Ex& ex, BOOST_THREAD_FWD_REF(F) f) {

  boost::shared_ptr<boost::detail::future_executor_shared_state<R> > h(
    new boost::detail::future_executor_shared_state<R>());

  h->init(ex, boost::forward<F>(f));

  return BOOST_THREAD_FUTURE<R>(h);
};
} // detail

#if defined(BOOST_THREAD_PROVIDES_INVOKE) &&                                  \
    !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATE) &&                             \
    !defined(BOOST_NO_CXX11_HDR_TUPLE)

#ifdef BOOST_THREAD_RVALUE_REFERENCES_DONT_MATCH_FUNCTION_PTR
template <typename Ex, typename R, typename... Ts>
BOOST_THREAD_FUTURE<R>
  async(
    Ex& ex, R(*f)(BOOST_THREAD_FWD_REF(Ts)...),
    BOOST_THREAD_FWD_REF(Ts) ...ts) {

  typedef R(*F)(BOOST_THREAD_FWD_REF(Ts)...);
  typedef boost::detail::invoker<
    typename boost::decay<F>::type,
    typename boost::decay<Ts>::type...> callback_type;
  typedef typename callback_type::result_type result_type;

  return BOOST_THREAD_MAKE_RV_REF(
    boost::detail::make_future_executor_shared_state<result_type>(
      ex,
      callback_type(
        boost::thread_detail::decay_copy(boost::forward<F>(f)),
        boost::thread_detail::decay_copy(boost::forward<Ts>(ts))...
      )));
}
#endif // BOOST_THREAD_RVALUE_REFERENCES_DONT_MATCH_FUNCTION_PTR
template <typename Ex, typename F, typename... Ts>
BOOST_THREAD_FUTURE<
  typename boost::result_of<
    typename boost::decay<F>::type(
      typename boost::decay<Ts>::type...)>::type>
  async(
    Ex& ex,
    BOOST_THREAD_FWD_REF(F) f,
    BOOST_THREAD_FWD_REF(Ts) ...ts) {

  typedef boost::detail::invoker<
    typename boost::decay<F>::type,
    typename boost::decay<Ts>::type...> callback_type;
  typedef typename callback_type::result_type result_type;

  return BOOST_THREAD_MAKE_RV_REF(
    boost::detail::make_future_executor_shared_state<result_type>(
      ex,
      callback_type(
        boost::thread_detail::decay_copy(boost::forward<F>(f)),
        boost::thread_detail::decay_copy(boost::forward<Ts>(ts))...
      )));
}
#else
#ifdef BOOST_THREAD_RVALUE_REFERENCES_DONT_MATCH_FUNCTION_PTR
template <typename Ex, typename R>
BOOST_THREAD_FUTURE<R>
  async(Ex& ex, R(*f)()) {

  typedef R(*F)();
  typedef boost::detail::invoker<F> callback_type;
  typedef typename callback_type::result_type result_type;

  return BOOST_THREAD_MAKE_RV_REF(
    boost::detail::make_future_executor_shared_state<result_type>(
      ex,
      callback_type(
        f)));
}

template <typename Ex, typename R, typename A>
BOOST_THREAD_FUTURE<R>
  async(
    Ex& ex,
    R(*f)(BOOST_THREAD_FWD_REF(A)),
    BOOST_THREAD_FWD_REF(A) a) {

  typedef R(*F)(BOOST_THREAD_FWD_REF(A));
  typedef boost::detail::invoker<
    F,
    typename boost::decay<A>::type> callback_type;
  typedef typename callback_type::result_type result_type;

  return BOOST_THREAD_MAKE_RV_REF(
    boost::detail::make_future_executor_shared_state<result_type>(
      ex,
      callback_type(
        f,
        boost::thread_detail::decay_copy(boost::forward<A>(a))
      )));
}
#endif // BOOST_THREAD_RVALUE_REFERENCES_DONT_MATCH_FUNCTION_PTR

template <typename Ex, typename F>
BOOST_THREAD_FUTURE<
  typename boost::result_of<
    typename boost::decay<F>::type()>::type>
  async(
    Ex& ex,
    BOOST_THREAD_FWD_REF(F) f) {

  typedef boost::detail::invoker<
    typename boost::decay<F>::type> callback_type;
  typedef typename callback_type::result_type result_type;

  return boost::detail::make_future_executor_shared_state<result_type>(
    ex,
    callback_type(
      boost::thread_detail::decay_copy(boost::forward<F>(f))));
}

template <typename Ex, typename F, typename T1>
BOOST_THREAD_FUTURE<
  typename boost::result_of<
    typename boost::decay<F>::type(
      typename boost::decay<T1>::type)>::type
  async(
    Ex& ex,
    BOOST_THREAD_FWD_REF(F) f,
    BOOST_THREAD_FWD_REF(T1) t1) {

  typedef boost::detail::invoker<
    typename boost::decay<F>::type,
    typename boost::decay<T1>::type> callback_type;
  typedef typename callback_type::result_type result_type;

  return BOOST_THREAD_MAKE_RV_REF(
    boost::detail::make_future_executor_shared_state<result_type>(
      ex,
      result_type(
        boost::thread_detail::decay_copy(boost::forward<F>(f)),
        boost::thread_detail::decay_copy(boost::forward<T1>(t1))));
}

template <typename Ex, typename F, typename T1, typename T2>
BOOST_THREAD_FUTURE<
  typename boost::result_of<
    typename boost::decay<F>::type(
      typename boost::decay<T1>::type,
      typename boost::decay<T2>::type)>::type>
  async(
    Ex& ex,
    BOOST_THREAD_FWD_REF(F) f,
    BOOST_THREAD_FWD_REF(T1) t1,
    BOOST_THREAD_FWD_REF(T2) t2) {

  typedef boost::detail::invoker<
    typename boost::decay<F>::type,
    typename boost::decay<T1>::type,
    typename boost::decay<T2>::type> callback_type;
  typedef typename callback_type::result_type result_type;

  return BOOST_THREAD_MAKE_RV_REF(
    boost::detail::make_future_executor_shared_state<result_type>(
      ex,
      boost::thread_detail::decay_copy(boost::forward<F>(f)),
      boost::thread_detail::decay_copy(boost::forward<T1>(t1)),
      boost::thread_detail::decay_copy(boost::forward<T2>(t2))));
}
#endif // BOOST_THREAD_PROVIDES_INVOKE
       // BOOST_NO_CXX11_VARIADIC_TEMPLATE
       // BOOST_NO_CXX11_HDR_TUPLE
#endif // BOOST_THREAD_PROVIDES_EXECUTORS

#ifdef BOOST_THREAD_RVALUE_REFERENCES_DONT_MATCH_FUNCTION_PTR
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
template <typename R, typename... Ts>
BOOST_THREAD_FUTURE<R>
  async(
    R(*f)(BOOST_THREAD_FWD_REF(Ts)...),
    BOOST_THREAD_FWD_REF(Ts) ...ts) {

  return BOOST_THREAD_MAKE_RV_REF(
    boost::async(
      boost::launch(
        boost::launch::any),
    f,
    boost::forward<Ts>(ts)...));
}
#else
template <typename R>
BOOST_THREAD_FUTURE<R>
  async(R(*f)()) {

  return BOOST_THREAD_MAKE_RV_REF(
    boost::sync(
      boost::launch(
        boost::launch::any),
      f));
}
#endif // BOOST_THREAD_RVALUE_REFERENCES_DONT_MATCH_FUNCTION_PTR
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD

#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
template <typename F, typename... Ts>
BOOST_THREAD_FUTURE<
  typename boost::result_of<
    typename boost::decay<F>::type(
      typename boost::decay<Ts>::type...)>::type>
  async(
    BOOST_THREAD_FWD_REF(F) f,
    BOOST_THREAD_FWD_REF(Ts) ...ts) {

  return BOOST_THREAD_MAKE_RV_REF(
    boost::async(
      boost::launch(
        boost::launch::any),
      boost::forward<F>(f),
      boost::forward<Ts>(ts)...));
}
#else
template <typename F>
BOOST_THREAD_FUTURE<
  typename boost::result_of<F()>::type>
  async(BOOST_THREAD_FWD_REF(F) f) {

  return BOOST_THREAD_MAKE_RV_REF(
    boost::async(
      boost::launch(
        boost::launch::any),
      boost::forward<F>(f)));
}
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD

/* make_future */
template <typename T>
BOOST_THREAD_FUTURE<typename boost::decay<T>::type>
  make_future(
    BOOST_THREAD_FWD_REF(T) value) {

  typedef typename boost::decay<T>::type future_value_type;
  boost::promise<future_value_type> p;
  p.set_value(boost::forward<future_value_type>(value));

  return BOOST_THREAD_MAKE_RV_REF(p.get_future());
}

#ifdef BOOST_THREAD_USES_MOVE
inline BOOST_THREAD_FUTURE<void> make_future() {

  boost::promise<void> p;
  p.set_value();

  return BOOST_THREAD_MAKE_RV_REF(p.get_future());
}
#endif // BOOST_THREAD_USES_MOVE

namespace detail {
/* make_ready_future */
template <typename T>
struct deduced_type_impl {
  typedef T type;
};

template <typename T>
struct deduced_type_impl<reference_wrapper<T> const> {
  typedef T& type;
};

template <typename T>
struct deduced_type_impl<reference_wrapper<T> > {
  typedef T& type;
};

template <typename T>
struct deduced_type_impl<std::reference_wrapper<T> > {
  typedef T& type;
};

template <typename T>
struct deduced_type {
  typedef typename boost::detail::deduced_type_impl<
    typename boost::decay<T>::type>::type type;
};
} // detail

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
template <int = 0, int..., typename T>
#else
template <typename T>
#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATE
BOOST_THREAD_FUTURE<typename boost::detail::deduced_type<T>::type>
  make_ready_future(BOOST_THREAD_FWD_REF(T) value) {

  typedef typename boost::detail::deduced_type<T>::type future_value_type;
  boost::promise<future_value_type> p;
  p.set_value(boost::forward<T>(value));

  return BOOST_THREAD_MAKE_RV_REF(p.get_future());
}

template <typename T>
BOOST_THREAD_FUTURE<T> make_ready_future(
  typename boost::remove_reference<T>::typ& v) {

  boost::promise<T> p;
  p.set_value(boost::forward<typename boost::remove_reference<T>::type>(v));

  return p.get_future();
}

template <typename T>
BOOST_THREAD_FUTURE<T> make_ready_future(
  BOOST_THREAD_FWD_REF(
    typename boost::remove_reference<T>::type) v) {
  boost::promise<T> p;
  p.set_value(boost::forward<typename boost::remove_reference<T>::type>(v));
  return p.get_future();
}

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
template <typename T, typename... Ts>
BOOST_THREAD_FUTURE<T> make_ready_future(Ts&& ...ts) {

  boost::promise<T> p;
  p.emplace(boost::forward<Ts>(ts)...);

  return p.get_future();
}
#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATES

template <typename T1, typename T2>
BOOST_THREAD_FUTURE<T1> make_ready_no_decay_future(T2 v) {
  typedef T1 future_value_type;
  boost::promise<future_value_type> p;
  p.set_value(v);
  return BOOST_THREAD_MAKE_RV_REF(p.get_future());
}

#if !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATE) ||                             \
    defined(BOOST_THREAD_USES_MOVE)
inline BOOST_THREAD_FUTURE<void> make_ready_future() {
  boost::promise<void> p;
  p.set_value();
  return p.get_future();
}
#endif

/* make_exceptional_future */
template <typename T>
BOOST_THREAD_FUTURE<T> make_exceptional_future(boost::exception_ptr e) {

  boost::promise<T> p;
  p.set_exception(boost::copy_exception(e));

  return BOOST_THREAD_MAKE_RV_REF(p.get_future());
}

template <typename T, typename E>
BOOST_THREAD_FUTURE<T> make_exceptional_future(E e) {

  boost::promise<T> p;
  p.set_exception(boost::copy_exception(e));

  return BOOST_THREAD_MAKE_RV_REF(p.get_future());
}

template <typename T>
BOOST_THREAD_FUTURE<T> make_exceptional_future() {

  boost::promise<T> p;
  p.set_exception(boost::current_exception());

  return BOOST_THREAD_MAKE_RV_REF(p.get_future());
}

template <typename T>
BOOST_THREAD_FUTURE<T> make_ready_future(boost::exception_ptr e) {
  return make_exceptional_future<T>(e);
}

/* make_shared_future */
template <typename T>
shared_future<typename boost::decay<T>::type>
  make_shared_future(BOOST_THREAD_FWD_REF(T) v) {

  typedef typename boost::decay<T>::type future_type;
  boost::promise<future_type> p;
  p.set_value(boost::forward<T>(v));

  return BOOST_THREAD_MAKE_RV_REF(p.get_future().share());
}

inline shared_future<void> make_shared_future() {
  boost::promise<void> p;
  return BOOST_THREAD_MAKE_RV_REF(p.get_future().share());
}

#if defined BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION

namespace detail {

template <
  typename F,
  typename R,
  typename C,
  typename S = boost::detail::shared_state<R> >
struct continuation_shared_state : S {
  F f_;
  C c_;

  continuation_shared_state(
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c) : f_(boost::move(f)), c_(boost::move(c)) {}

  void init(boost::unique_lock<boost::mutex>& lock) {
    f_.future_->set_continuation_ptr(this->shared_from_this(), lock);
  }

  void call() {

   try {
     this->mark_finished_with_result(
       this->c_(boost::move(this->f_)));
   } catch (...) {
     this->mark_exceptional_finish();
   }
   this->f_ = F();
  }

  void call(boost::unique_lock<boost::mutex>& lock) {

    try {
      relocker relock(lock);
      R r = this->c_(boost::move(this->f_));
      this->f_ = F();
      relock.lock();
      this->mark_finished_with_result_internal(boost::move(r), lock);
    } catch (...) {
      this->mark_exceptional_finish_internal(
        boost::current_exception(), lock);
      relocker relock(lock);
      this->f_ = F();
    }
  }

  static void run(boost::shared_ptr<boost::detail::shared_state_base> that) {

    continuation_shared_state* that_ =
      static_cast<continuation_shared_state*>(that.get());
    that_->call();
  }

  ~continuation_shared_state() {}
};

template <typename F, typename C, typename S>
struct continuation_shared_state<F, void, C, S> : S {
  F f_;
  C c_;

  continuation_shared_state(
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_RV_REF(C) c) : f_(boost::move(f)), c_(boost::move(c)) {}

  void init(boost::unique_lock<boost::mutex>& lock) {
    f_.future_->set_continuation_ptr(this->shared_from_this(), lock);
  }

  void call() {

    try {
      this->c_(boost::move(this->f_));
      this->mark_finished_with_result();
    } catch (...) {
      this->mark_exceptional_finish();
    }
    this->f_ = F();
  }

  void call(boost::unique_lock<boost::mutex>& lock) {

    try {
      relocker relock(lock);
      this->c_(boost::move(this->f_));
      this->f_ = F();
    } catch (...) {
      relocker relock(lock);
      this->f_ = F();
    }
  }

  static void run(boost::shared_ptr<boost::detail::shared_state_base> that) {

    continuation_shared_state* that_ =
      static_cast<continuation_shared_state*>(that.get());
    that_->call();
  }

  ~continuation_shared_state() {}
};

/* future_async_continuation_shared_state */
template <typename F, typename R, typename C>
struct future_async_continuation_shared_state :
  boost::detail::continuation_shared_state<
    F, R, C, boost::detail::future_async_shared_state_base<R> > {
  typedef boost::detail::continuation_shared_state<
    F, R, C, boost::detail::future_async_shared_state_base<R> > base_type;

  future_async_continuation_shared_state(
    BOOST_THREAD_RV_REF(F) f, BOOST_THREAD_FWD_REF(C) c) :
    base_type(boost::move(f), boost::forward<C>(c)) {}

  void launch_continuation() {
#ifdef BOOST_THREAD_FUTURE_BLOCKING
    boost::lock_guard<boost::mutex> lock(this->mutex_);
    this->thr_ =
      boost::thread(&future_async_continuation_shared_state::run,
        static_shared_from_this(this));
#else
    boost::thread(&base_type::run, static_shared_from_this(this)).detach();
#endif // BOOST_THREAD_FUTURE_BLOCKING
  }
};

/* future_sync_continuation_shared_state */
template <typename F, typename R, typename C>
struct future_sync_continuation_shared_state :
  boost::detail::continuation_shared_state<
    F, R, C, boost::detail::shared_state<R> >{
  typedef boost::detail::continuation_shared_state<
    F, R, C, boost::detail::shared_state<R> > base_type;

  future_sync_continuation_shared_state(
    BOOST_THREAD_RV_REF(F) f, BOOST_THREAD_FWD_REF(C) c) :
    base_type(boost::move(f), boost::forward<C>(c)) {}

  void launch_continuation() {
    this->call();
  }

};

/* future_async_continuation_shared_state */
#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
template <typename S>
struct run_it {
  boost::shared_ptr<S> ex_;

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
  BOOST_THREAD_COPYABLE_AND_MOVABLE(run_it)
  run_it(run_it const& ex) : ex_(ex.ex_) {}

  run_it& operator=(BOOST_THREAD_COPY_ASSIGN_REF(run_it) ex) {

    if (this != &ex) {
      ex_ = ex.ex_;
    }
    return *this;
  }

  run_it(BOOST_THREAD_RV_REF(run_it) ex) BOOST_NOEXCEPT : ex_(ex.ex_) {
    ex.ex_.reset();
  }

  run_it& operator=(BOOST_THREAD_RV_REF(run_it) ex) BOOST_NOEXCEPT {

    if (this != &ex) {
      ex_ = ex.ex_;
      ex.ex_.reset();
    }
    return *this;
  }
#endif // BOSOT_NO_CXX11_RVALUE_REFERENCES

  run_it(boost::shared_ptr<S> ex) : ex_(ex) {}

  void operator()() {
    ex_->run(ex_);
  }
};
} // detail

BOOST_THREAD_DCL_MOVABLE_BEG(F)
boost::detail::run_it<F>
BOOST_THREAD_DCL_MOVABLE_END

namespace detail {

template <typename F, typename R, typename C>
struct future_executor_continuation_shared_state :
  boost::detail::continuation_shared_state<F, R, C> {
  typedef boost::detail::continuation_shared_state<F, R, C> base_type;

  future_executor_continuation_shared_state(
    BOOST_THREAD_RV_REF(F) f, BOOST_THREAD_RV_REF(C) c) :
    base_type(boost::move(f), boost::forward<C>(c)) {}

  template <typename Ex>
  void init(boost::unique_lock<boost::mutex>& lock, Ex ex) {

    this->set_executor_policy(boost::executor_ptr_type(
      new executor_ref<Ex>(ex)), lock);
    this->base_type::init(lock);
  }

  void launch_continuation() {

    boost::detail::run_it<base_type> f(static_shared_from_this(this));
    this->get_executor()->submit(boost::move(f));
  }

  ~future_executor_continuation_shared_state() {}
};
#endif // BOOST_THREAD_PROVIDES_EXECUTORS

/* shared_future_async_continuation_shared_state */
template <typename F, typename R, typename C>
struct shared_future_async_continuation_shared_state :
  boost::detail::continuation_shared_state<
    F, R, C, boost::detail::future_async_shared_state_base<R> > {
  typedef boost::detail::continuation_shared_state<
    F, R, C, boost::detail::future_async_shared_state_base<R> > base_type;

  shared_future_async_continuation_shared_state(
    F f, BOOST_THREAD_FWD_REF(C) c) :
    base_type(boost::move(f), boost::forward<C>(c)) {}

  void launch_continuation() {
#ifdef BOOST_THREAD_FUTURE_BLOCKING
    boost::lock_guard<boost::mutex> lock(this->mutex_);
    this->thr_ =
      boost::thread(&base_type::run, static_shared_from_this(this));
#else
    boost::thread(&base_type::run, static_shared_from_this(this)).detach();
#endif // BOOST_THREAD_FUTURE_BLOCKING
  }
};

/* shared_future_sync_continuation_shared_state */
template <typename F, typename R, typename C>
struct shared_future_sync_continuation_shared_state :
  boost::detail::continuation_shared_state<
    F, R, C, boost::detail::shared_state<R> > {
  typedef boost::detail::continuation_shared_state<
    F, R, C, boost::detail::shared_state<R> > base_type;

  shared_future_sync_continuation_shared_state(
    F f, BOOST_THREAD_FWD_REF(C) c) :
    base_type(boost::move(f), boost::forward<C>(c)) {}

  void launch_continuation() {
    this->call();
  }
};

/*  shared_future_executor_continuation_shared_state */
#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
template <typename F, typename R, typename C>
struct shared_future_executor_continuation_shared_state :
  boost::detail::continuation_shared_state<F, R, C> {
  typedef boost::detail::continuation_shared_state<F, R, C> base_type;

  shared_future_executor_continuation_shared_state(
    F f, BOOST_THREAD_FWD_REF(C) c) :
    base_type(boost::move(f), boost::forward<C>(c)) {}

  template <typename Ex>
  void init(boost::unique_lock<boost::mutex>& lock, Ex& ex) {
    this->set_executor_policy(*boost::executor_ptr_type(
      new executor_ref<Ex>(ex)), lock);
  }

  void launch_continuation() {
    boost::detail::run_it<base_type> f(static_shared_from_this(this));
    this->get_executor()->submit(boost::move(f));
  }

  ~shared_future_executor_continuation_shared_state() {}
};
#endif // BOOST_THREAD_PROVIDES_EXECUTORS

/* future_deferred_continuation_shared_state */
template <typename F, typename R, typename C>
struct future_deferred_continuation_shared_state :
  boost::detail::continuation_shared_state<F, R, C> {
  typedef boost::detail::continuation_shared_state<F, R, C> base_type;

  future_deferred_continuation_shared_state(
    BOOST_THREAD_RV_REF(F) f, BOOST_THREAD_FWD_REF(C) c) :
    base_type(boost::move(f), boost::forward<C>(c)) {
    this->set_deferred();
  }

  virtual void execute(boost::unique_lock<boost::mutex>& lock) {
    this->f_.wait();
    this->call(lock);
  }

  virtual void launch_continuation() {}
};

/* shared_future_deferred_continuation_shared_state */
template <typename F, typename R, typename C>
struct shared_future_deferred_continuation_shared_state :
  boost::detail::continuation_shared_state<F, R, C> {
  typedef boost::detail::continuation_shared_state<F, R, C> base_type;

  shared_future_deferred_continuation_shared_state(
    F f, BOOST_THREAD_FWD_REF(C) c) :
    base_type(boost::move(f), boost::forward<C>(c)) {
    this->set_deferred();
  }

  virtual void execute(boost::unique_lock<boost::mutex>& lock) {
    this->f_.wait();
    this->call(lock);
  }

  virtual void launch_continuation() {}
};

/* make_future_async_continuation_shared_state */
template <typename F, typename R, typename C>
BOOST_THREAD_FUTURE<R>
  make_future_async_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c) {

  typedef typename boost::decay<C>::type callback_type;
  boost::shared_ptr<
    boost::detail::future_async_continuation_shared_state<
      F, R, callback_type> > h(
        new boost::detail::future_async_continuation_shared_state<
          F, R, callback_type>(boost::move(f), boost::forward<C>(c)));
  h->init(lock);

  return BOOST_THREAD_FUTURE<R>(h);
}

/* make_future_sync_continuaiton_shared_state */
template <typename F, typename R, typename C>
BOOST_THREAD_FUTURE<R>
  make_future_sync_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c) {

  typedef typename boost::decay<C>::type callback_type;
  boost::shared_ptr<
    boost::detail::future_sync_continuation_shared_state<
      F, R, callback_type> > h(
        new boost::detail::future_sync_continuation_shared_state<
          F, R, callback_type>(boost::move(f), boost::forward<C>(c)));
  h->init(lock);

  return BOOST_THREAD_FUTURE<R>(h);
}

/* make_future_deferred_continuation_shared_state */
template <typename F, typename R, typename C>
BOOST_THREAD_FUTURE<R>
  make_future_deferred_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c) {

  typedef typename boost::decay<C>::type callback_type;
  boost::shared_ptr<
    boost::detail::future_deferred_continuation_shared_state<
      F, R, callback_type> > h(
        new boost::detail::future_deferred_continuation_shared_state<
          F, R, callback_type>(boost::move(f), boost::forward<C>(c)));

  h->init(lock);

  return BOOST_THREAD_FUTURE<R>(h);
}

/* make_future_executor_continuation_shared_state */
#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
template <typename Ex, typename F, typename R, typename C>
BOOST_THREAD_FUTURE<R>
  make_future_executor_continuation_shared_state(
    Ex& ex,
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c) {
  typedef typename boost::decay<C>::type callback_type;
  boost::shared_ptr<
    boost::detail::future_executor_continuation_shared_state<
      F, R, callback_type> > h(
        new boost::detail::future_executor_continuation_shared_state<
          F, R, callback_type>(boost::move(f), boost::forward<C>(c)));
  h->init(lock, ex);

  return BOOST_THREAD_FUTURE<R>(h);
}
#endif

/* make_shared_future_async_continuation_shared_state */
template <typename F, typename R, typename C>
BOOST_THREAD_FUTURE<R>
  make_shared_future_async_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    F f,
    BOOST_THREAD_FWD_REF(C) c) {
  typedef typename boost::decay<C>::type callback_type;
  boost::shared_ptr<
    boost::detail::shared_future_async_continuation_shared_state<
      F, R, callback_type> > h(
        new boost::detail::shared_future_async_continuation_shared_state<
          F, R, callback_type>(f, boost::forward<C>(c)));
  h->init(lock);

  return BOOST_THREAD_FUTURE<R>(h);
}

/* make_shared_future_sync_continuation_shared_state */
template <typename F, typename R, typename C>
BOOST_THREAD_FUTURE<R>
  make_shared_future_sync_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    F f,
    BOOST_THREAD_FWD_REF(C) c) {
  typedef typename boost::decay<C>::type callback_type;
  boost::shared_ptr<
    boost::detail::shared_future_sync_continuation_shared_state<
      F, R, callback_type> > h(
        new boost::detail::shared_future_sync_continuation_shared_state<
          F, R, callback_type>(f, boost::forward<C>(c)));
  h->init(lock);

  return BOOST_THREAD_FUTURE<R>(h);
}

/* make_shared_future_deferred_continuation_shared_state */
template <typename F, typename R, typename C>
BOOST_THREAD_FUTURE<R>
  make_shared_future_deferred_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    F f,
    BOOST_THREAD_FWD_REF(C) c) {
  typedef typename boost::decay<C>::type callback_type;
  boost::shared_ptr<
    boost::detail::shared_future_deferred_continuation_shared_state<
      F, R, callback_type> > h(
        new boost::detail::shared_future_deferred_continuation_shared_state<
          F, R, callback_type>(f, boost::forward<C>(c)));
  h->init(lock);

  return BOOST_THREAD_FUTURE<R>(h);
}
} // detail

template <typename R>
template <typename F>
inline BOOST_THREAD_FUTURE<
  typename boost::result_of<
    F(BOOST_THREAD_FUTURE<R>)>::type>
      BOOST_THREAD_FUTURE<R>::then(
        boost::launch policy,
        BOOST_THREAD_FWD_REF(F) f) {

  typedef typename boost::result_of<
    F(BOOST_THREAD_FUTURE<R>)>::type future_type;

  BOOST_THREAD_ASSERT_PRECONDITION(
    this->future_.get() != 0,
    boost::future_uninitialized());

  boost::shared_ptr<
    boost::detail::shared_state_base> shared_state(this->future_);
  boost::unique_lock<boost::mutex> lock(shared_state->mutex_);

  if (boost::underlying_cast<int>(policy) &&
      int(boost::launch::async)) {

    return BOOST_THREAD_MAKE_RV_REF((
      boost::detail::make_future_async_continuation_shared_state<
        BOOST_THREAD_FUTURE<R>, future_type>(
          lock, boost::move(*this), boost::forward<F>(f))));

  } else if (boost::underlying_cast<int>(policy) &&
             int(boost::launch::deferred)) {

    return BOOST_THREAD_MAKE_RV_REF((
      boost::detail::make_future_deferred_continuation_shared_state<
        BOOST_THREAD_FUTURE<R>, future_type>(
          lock, boost::move(*this), boost::forward<F>(f))));

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
  } else if (boost::underlying_cast<int>(policy) &&
             int(boost::launch::executor)) {

    assert(this->future_->get_executor());
    typedef boost::executor Ex;
    Ex& ex = *(this->future_->get_executor());

    return BOOST_THREAD_MAKE_RV_REF((
      boost::detail::make_future_executor_continuation_shared_state<
        Ex, BOOST_THREAD_FUTURE<R>, future_type>(
          ex, lock, boost::move(*this), boost::forward<F>(f))));

#endif // BOOST_THREAD_PROVIDES_EXECUTORS
  } else if (boost::underlying_cast<int>(policy) &&
             int(boost::launch::inherit)) {

    boost::launch policy_ = this->launch_policy(lock);

    if (boost::underlying_cast<int>(policy_) &&
        int(boost::launch::async)) {

      return BOOST_THREAD_MAKE_RV_REF((
        boost::detail::make_future_async_continuation_shared_state<
          BOOST_THREAD_FUTURE<R>, future_type>(
            lock, boost::move(*this), boost::forward<F>(f))));

    } else if (boost::underlying_cast<int>(policy) &&
               int(boost::launch::deferred)) {

      return BOOST_THREAD_MAKE_RV_REF((
        boost::detail::make_future_deferred_continuation_shared_state<
          BOOST_THREAD_FUTURE<R>, future_type>(
            lock, boost::move(*this), boost::forward<F>(f))));

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
    } else if (boost::underlying_cast<int>(policy) &&
               int(boost::launch::deferred)) {

      assert(this->future_->get_executor());
      typedef boost::executor Ex;
      Ex& ex = *(this->future_->get_executor());

      return BOOST_THREAD_MAKE_RV_REF((
        boost::detail::make_future_executor_continuation_shared_state<
          BOOST_THREAD_FUTURE<R>, future_type>(
            lock, boost::move(*this), boost::forward<F>(f))));
#endif // BOOST_THREAD_PROVIDES_EXECUTORS
    } else {

      return BOOST_THREAD_MAKE_RV_REF((
        boost::detail::make_future_async_continuation_shared_state<
          BOOST_THREAD_FUTURE<R>, future_type>(
            lock, boost::move(*this), boost::forward<F>(f))));

    }
  } else {

    return BOOST_THREAD_MAKE_RV_REF((
      boost::detail::make_future_async_continuation_shared_state<
        BOOST_THREAD_FUTURE<R>, future_type>(
          lock, boost::move(*this), boost::forward<F>(f))));
  }
}

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
template <typename R>
template <typename Ex, typename F>
inline BOOST_THREAD_FUTURE<
  typename boost::result_of<
    F(BOOST_THREAD_FUTURE<R>)>::type>
     BOOST_THREAD_FUTURE<R>::then(
       Ex& ex,
       BOOST_THREAD_FWD_REF(F) f) {
  typedef typename boost::result_of<
    F(BOOST_THREAD_FUTURE<R>)>::type future_type;

  BOOST_THREAD_ASSERT_PRECONDITION(
    this->future_.get() != 0,
    boost::future_uninitialized());

  boost::shared_ptr<
    boost::detail::shared_state_base> shared_state(this->future_);
  boost::unique_lock<boost::mutex> lock(shared_state->mutex_);

  return BOOST_THREAD_MAKE_RV_REF((
    boost::detail::make_future_executor_continuation_shared_state<
      Ex, BOOST_THREAD_FUTURE<R>, future_type>(
        ex, lock, boost::move(*this), boost::forward<F>(f))));
}
#endif // BOOST_THREAD_PROVIDES_EXECUTORS

template <typename R>
template <typename F>
inline BOOST_THREAD_FUTURE<
  typename boost::result_of<
    F(BOOST_THREAD_FUTURE<R>)>::type>
      BOOST_THREAD_FUTURE<R>::then(
        BOOST_THREAD_FWD_REF(F) f) {
#ifndef BOOST_THREAD_CONTINUATION_SYNC
  return this->then(this->launch_policy(), boost::forward<F>(f));
#else
  typedef typename boost::result_of<
    F(BOOST_THREAD_FUTURE<R>)>::type future_type;
  BOOST_THREAD_ASSERT_PRECONDITION(
    this->future_.get() != 0,
    boost::future_uninitialized());

  boost::shared_ptr<
    boost::detail::shared_state_base> shared_state(this->future_);
  boost::unique_lock<boost::mutex> lock(shared_state->mutex_);
  boost::launch policy = this->launch_policy(lock);

  if (boost::underlying_cast<int>(policy) &&
      int(boost::launch::deferred)) {

    return BOOST_THREAD_MAKE_RV_REF((
      boost::detail::make_future_deferred_continuation_shared_state<
        BOOST_THREAD_FUTURE<R>, future_type>(
          lock, boost::move(*this), boost::forward<F>(f))));

  } else {

    return BOOST_THREAD_MAKE_RV_REF((
      boost::detail::make_future_async_continuation_shared_state<
        BOOST_THREAD_FUTURE<R>, future_type>(
          lock, boost::move(*this), boost::forward<F>(f))));
  }
#endif
}

template <typename R>
template <typename F>
inline BOOST_THREAD_FUTURE<
  typename boost::result_of<
    F(BOOST_THREAD_FUTURE<BOOST_THREAD_FUTURE<R> >)>::type>
      BOOST_THREAD_FUTURE<BOOST_THREAD_FUTURE<R> >::then(
        boost::launch policy,
        BOOST_THREAD_FWD_REF(F) f) {
  typedef BOOST_THREAD_FUTURE<R> R2;
  typedef typename boost::result_of<
    F(BOOST_THREAD_FUTURE<R2>)>::type future_type;

  BOOST_THREAD_ASSERT_PRECONDITION(
    this->future_.get() != 0,
    boost::future_uninitialized());

  boost::shared_ptr<
    boost::detail::shared_state_base> shared_state(this->future_);
  boost::unique_lock<boost::mutex> lock(shared_state->mutex_);

  if (boost::underlying_cast<int>(policy) &&
      int(boost::launch::async)) {

    return BOOST_THREAD_MAKE_RV_REF((
      boost::detail::make_future_async_continuation_shared_state<
        BOOST_THREAD_FUTURE<R2>, future_type>(
          lock, boost::move(*this), boost::forward<F>(f))));

  } else if (boost::underlying_cast<int>(policy) &&
             int(boost::launch::sync)) {

    return BOOST_THREAD_MAKE_RV_REF((
      boost::detail::make_future_sync_continuation_shared_state<
        BOOST_THREAD_FUTURE<R2>, future_type>(
          lock, boost::move(*this), boost::forward<F>(f))));

  } else if (boost::underlying_cast<int>(policy) &&
             int(boost::launch::deferred)) {

    return BOOST_THREAD_MAKE_RV_REF((
      boost::detail::make_future_deferred_continuation_shared_state<
        BOOST_THREAD_FUTURE<R2>, future_type>(
          lock, boost::move(*this), boost::forward<F>(f))));

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
  } else if (boost::underlying_cast<int>(policy) &&
             int(boost::launch::executor)) {

    assert(this->future_->get_executor());
    typedef executor Ex;
    Ex& ex = *(this->future_->get_executor());

    return BOOST_THREAD_MAKE_RV_REF((
      boost::detail::make_future_executor_continuation_shared_state<
        Ex, BOOST_THREAD_FUTURE<R2>, future_type>(
          ex, lock, boost::move(*this), boost::forward<F>(f))));
#endif // BOOST_THREAD_PROVIDES_EXECUTORS

  } else if (boost::underlying_cast<int>(policy) &&
             int(boost::launch::inherit)) {

    boost::launch policy_ = this->launch_policy(lock);

    if (boost::underlying_cast<int>(policy_) &&
        int(boost::launch::async)) {

      return BOOST_THREAD_MAKE_RV_REF((
        boost::detail::make_future_async_continuation_shared_state<
          BOOST_THREAD_FUTURE<R2>, future_type>(
            lock, boost::move(*this), boost::forward<F>(f))));

    } else if (boost::underlying_cast<int>(policy_) &&
               int(boost::launch::sync)) {

      return BOOST_THREAD_MAKE_RV_REF((
        boost::detail::make_future_sync_continuation_shared_state<
          BOOST_THREAD_FUTURE<R2>, future_type>(
            lock, boost::move(*this), boost::forward<F>(f))));

    } else if (boost::underlying_cast<int>(policy_) &&
               int(boost::launch::deferred)) {

      return BOOST_THREAD_MAKE_RV_REF((
        boost::detail::make_future_deferred_continuation_shared_state<
          BOOST_THREAD_FUTURE<R2>, future_type>(
            lock, boost::move(*this), boost::forward<F>(f))));

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
    } else if (boost::underlying_cast<int>(policy_) &&
               int(boost::launch::executor)) {

      assert(this->future_->get_executor());
      typedef boost::executor Ex;
      Ex& ex = *(this->future_->get_executor());

      return BOOST_THREAD_MAKE_RV_REF((
        boost::detail::make_future_executor_continuation_shared_state<
          Ex, BOOST_THREAD_FUTURE<R2>, future_type>(
            ex, lock, boost::move(*this), boost::forward<F>(f))));
#endif // BOOST_THREAD_PROVIDES_EXECUTORS

    } else {

      return BOOST_THREAD_MAKE_RV_REF((
        boost::detail::make_future_async_continuation_shared_state<
          BOOST_THREAD_FUTURE<R2>, future_type>(
            lock, boost::move(*this), boost::forward<F>(f))));
    }
  } else {

    return BOOST_THREAD_MAKE_RV_REF((
      boost::detail::make_future_async_continuation_shared_state<
        BOOST_THREAD_FUTURE<R2>, future_type>(
          lock, boost::move(*this), boost::forward<F>(f))));

  }
}

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
template <typename R>
template <typename Ex, typename F>
inline BOOST_THREAD_FUTURE<
  typename boost::result_of<
    F(BOOST_THREAD_FUTURE<BOOST_THREAD_FUTURE<R> >)>::type>
      BOOST_THREAD_FUTURE<BOOST_THREAD_FUTURE<R> >::then(
        Ex& ex,
        BOOST_THREAD_FWD_REF(F) f) {
  typedef BOOST_THREAD_FUTURE<R> R2;
  typedef typename boost::result_of<
    F(BOOST_THREAD_FUTURE<R2>)>::type future_type;
  BOOST_THREAD_ASSERT_PRECONDITION(
    this->future_.get() != 0,
    boost::future_uninitialized());

  boost::shared_ptr<
    boost::detail::shared_state_base> shared_state(this->future_);
  boost::unique_lock<boost::mutex> lock(shared_state->mutex_);

  boost::launch policy = this->launch_policy(lock);

  if (boost::underlying_cast<int>(policy) &&
      int(boost::launch::deferred)) {

    return BOOST_THREAD_MAKE_RV_REF((
      boost::detail::make_future_deferred_continuation_shared_state<
        BOOST_THREAD_FUTURE<R2>, future_type>(
          lock, boost::move(*this), boost::forward<F>(f))));

  } else {

    return BOOST_THREAD_MAKE_RV_REF((
      boost::detail::make_future_sync_continuation_shared_state<
        BOOST_THREAD_FUTURE<R2>, future_type>(
          lock, boost::move(*this), boost::forward<F>(f))));
  }
#endif // BOOST_THREAD_PROVIDES_EXECUTORS
}

template <typename R>
template <typename F>
inline BOOST_THREAD_FUTURE<
  typename boost::result_of<
    F(BOOST_THREAD_FUTURE<BOOST_THREAD_FUTURE<R> >)>::type>
      BOOST_THREAD_FUTURE<BOOST_THREAD_FUTURE<R> >::then(
        BOOST_THREAD_FWD_REF(F) f) {
#ifndef BOOST_THREAD_CONTINUATION_SYNC
  return this->then(this->launch_policy(), boost::forward<F>(f));
#else
  typedef BOOST_THREAD_FUTURE<R> R2;
  typedef typename boost::result_of<
    F(BOOST_THREAD_FUTURE<R2>)>::type future_type;
  BOOST_ASSERT_PRECONDITION(
    this->future_.get() != 0,
    boost::future_uninitialized());

  boost::shared_ptr<
    boost::detail::shared_state_base> shared_state(this->future_);
  boost::unique_lock<boost::mutex> lock(shared_state->mutex_);

  boost::launch policy = this->launch_policy();

  if (boost::underlying_cast<int>(policy) &&
      int(boost::launch::deferred)) {

    return BOOST_THREAD_MAKE_RV_REF((
      boost::detail::make_future_deferred_continuation_shared_state<
        BOOST_THREAD_FUTURE<R2>, future_type>(
          lock, boost::move(*this), boost::forward<F>(f))));

  } else {

    return BOOST_THREAD_MAKE_RV_REF((
      boost::detail::make_future_sync_continuation_shared_state<
        BOOST_THREAD_FUTURE<R2>, future_type>(
          lock, boost::move(*this), boost::forward<F>(f))));
  }
#endif // BOOST_THREAD_CONTINUATION_SYNC
}

template <typename R>
template <typename F>
inline BOOST_THREAD_FUTURE<
  typename boost::result_of<
    F(boost::shared_future<R>)>::type>
      boost::shared_future<R>::then(
        boost::launch policy, BOOST_THREAD_FWD_REF(F) f) const {
  typedef typename boost::result_of<
    boost::shared_future<R> >::type future_type;
  BOOST_THREAD_ASSERT_PRECONDITION(
    this->future_.get() != 0,
    boost::future_uninitialized());

  boost::unique_lock<boost::mutex> lock(this->future_->mutex_);

  if (boost::underlying_cast<int>(policy) &&
      int(boost::launch::async)) {

    return BOOST_THREAD_MAKE_RV_REF((
      boost::detail::make_shared_future_async_continuation_shared_state<
        boost::shared_future<R>, future_type>(
          lock, *this, boost::forward<F>(f))));

  } else if (boost::underlying_cast<int>(policy) &&
             int(boost::launch::sync)) {

    return BOOST_THREAD_MAKE_RV_REF((
      boost::detail::make_shared_future_sync_continuation_shared_state<
        boost::shared_future<R>, future_type>(
          lock, *this, boost::forward<F>(f))));

  } else if (boost::underlying_cast<int>(policy) &&
             int(boost::launch::deferred)) {

    return BOOST_THREAD_MAKE_RV_REF((
      boost::detail::make_shared_future_deferred_continuation_shared_state<
        boost::shared_future<R>, future_type>(
          lock, *this, boost::forward<F>(f))));

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
  } else if (boost::underlying_cast<int>(policy) &&
             int(boost::launch::executor)) {

    assert(this->future_->get_executor());
    typedef boost::executor Ex;
    Ex& ex = *(this->future_->get_executor());

    return BOOST_THREAD_MAKE_FWD_REF((
      boost::detail::make_shared_future_executor_continuation_shared_state<
        Ex, boost::shared_future<R>, future_type>(
          ex, lock, *this, boost::forward<F>(f))));
#endif // BOOST_THREAD_PROVIDES_EXECUTORS
  } else if (boost::underlying_cast<int>(policy) &&
             int(boost::launch::inherit)) {

    boost::launch policy_ = this->launch_policy();

    if (boost::underlying_cast<int>(policy_) &&
        int(boost::launch::async)) {

      return BOOST_THREAD_MAKE_FWD_REF((
        boost::detail::make_shared_future_async_continuation_shared_state<
          boost::shared_future<R>, future_type>(
            lock, *this, boost::forward<F>(f))));

    } else if (boost::underlying_cast<int>(policy_) &&
              int(boost::launch::sync)) {

      return BOOST_THREAD_MAKE_FWD_REF((
        boost::detail::make_shared_future_sync_continuation_shared_state<
          boost::shared_future<R>, future_type>(
            lock, *this, boost::forward<F>(f))));

    } else if (boost::underlying_cast<int>(policy_) &&
               int(boost::launch::deferred)) {

      return BOOST_THREAD_MAKE_FWD_REF((
        boost::detail::make_shared_future_deferred_continuation_shared_state<
          boost::shared_future<R>, future_type>(
            lock, *this, boost::forward<F>(f))));

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
    } else if (boost::underlying_cast<int>(policy_) &&
               int(boost::launch::deferred)) {

      assert(this->future_->get_executor());
      typedef boost::executor Ex;
      Ex& ex = *(this->future_->get_executor());

      return BOOST_THREAD_MAKE_RV_REF((
        boost::detail::make_shared_future_executor_continuation_shared_state<
          Ex, boost::shared_future<R>, future_type>(
            ex, lock, *this, boost::forward<F>(f))));
#endif // BOOST_THREAD_PROVIDES_EXECUTORS

    } else {

      return BOOST_THREAD_MAKE_RV_REF((
        boost::detail::make_shared_future_async_continuation_shared_state<
          boost::shared_future<R>, future_type>(
            lock, *this, boost::forward<F>(f))));

    }
  } else {

    return BOOST_THREAD_MAKE_RV_REF((
      boost::detail::make_shared_future_async_continuation_shared_state<
        boost::shared_future<R>, future_type>(
          lock, *this, boost::forward<F>(f))));
  }
}

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
template <typename R>
template <typename Ex, typename F>
inline BOOST_THREAD_FUTURE<
  typename boost::result_of<
    F(boost::shared_future<R>)>::type>
      boost::shared_future<R>::then(
        Ex& ex, BOOST_THREAD_FWD_REF(F) f) const {
  typedef typename boost::result_of<
    F(boost::shared_future<R>)>::type future_type;
  BOOST_THREAD_ASSERT_PRECONDITION(
    this->future_.get() != 0,
    boost::future_uninitialized());

  boost::unique_lock<boost::mutex> lock(this->future_->mutex_);

  return BOOST_THREAD_MAKE_RV_REF((
    boost::detail::make_shared_future_executor_continuation_shared_state<
      Ex, shared_future<R>, future_type>(
        ex, lock, *this, boost::forward<F>(f))));
}
#endif // BOOST_THREAD_PROVIDES_EXECUTORS

template <typename R>
template <typename F>
inline BOOST_THREAD_FUTURE<
  typename boost::result_of<
    F(boost::shared_future<R>)>::type>
      boost::shared_future<R>::then(
        BOOST_THREAD_FWD_REF(F) f) const {
#ifndef BOOST_THREAD_CONTINUATION_SYNC
  return this->then(this->launch_policy(), boost::forward<F>(f));
#else
  typedef typename boost::result_of<
    F(boost::shared_future<R>)>::type future_type;
  BOOST_THREAD_ASSERT_PRECONDITION(
    this->future_.get() != 0,
    boost::future_uninitialized());

  boost::unique_lock<boost::mutex> lock(this->future_->mutex_);
  boost::launch policy = this->launch_policy();

  if (boost::underlying_cast<int>(policy) &&
      int(boost::launch::deferred)) {

    return BOOST_THREAD_MAKE_RV_REF((
      boost::detail::make_shared_future_deferred_continuation_shared_state<
        boost::shared_future<R>, future_type>(
          lock, *this, boost::forward<F>(f))));

  } else {

    return BOOST_THREAD_MAKE_RV_REF((
      boost::detail::make_shared_future_async_continuation_shared_state<
        boost::shared_future<R>, future_type>(
          lock, *this, boost::forward<F>(f))));

  }
#endif // BOOST_THREAD_CONTINUATION_SYNC
}

namespace detail {

/* move */
template <typename T>
struct mfallbacker_to {
  T value_;
  typedef T result_type;
  mfallbacker_to(BOOST_THREAD_RV_REF(T) value) : value_(boost::move(value)) {}

  T operator()(BOOST_THREAD_FUTURE<T> f) {
    return f.get_or(boost::move(value_));
  }
};

/* const */
template <typename T>
struct cfallbacker_to {
  T value_;
  typedef T result_type;
  cfallbacker_to(T const& value) : value_(value) {}

  T operator()(BOOST_THREAD_FUTURE<T> f) const {
    return f.get_or(value_);
  }
};
} // detail

template <typename R>
template <typename R2>
inline typename boost::disable_if<
  boost::is_void<R2>,
  BOOST_THREAD_FUTURE<R> >::type
    BOOST_THREAD_FUTURE<R>::fallback_to(BOOST_THREAD_RV_REF(R2) v) {
  return then(boost::detail::mfallbacker_to<R>(boost::move(v)));
}

template <typename R>
template <typename R2>
inline typename boost::disable_if<
  boost::is_void<R2>,
  BOOST_THREAD_FUTURE<R> >::type
    BOOST_THREAD_FUTURE<R>::fallback_to(R2 const& v) {
  return then(boost::detail::cfallbacker_to<R>(v));    
}

#endif // BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION

#ifdef BOOST_THREAD_PROVIDES_FUTURE_UNWRAP
namespace detail {

template <typename F, typename R>
struct future_unwrap_shared_state : boost::detail::shared_state<R> {

  F wrapped_;
  typename F::value_type unwrapped_;

  explicit future_unwrap_shared_state(BOOST_THREAD_RV_REF(F) f) :
    wrapped_(boost::move(f)) {}

  void launch_continuation() {
    boost::unique_lock<boost::mutex> lock(this->future_->mutex_);

    if (!unwrapped_.valid()) {
      if (unwrapped_.has_exception()) {
        this->mark_exceptional_finish_internal(
          wrapped_.get_exception_ptr(), lock);
      } else {
        unwrapped_ = wrapped_.get();
        if (unwrapped_.valid()) {
          lock.unlock();
          boost::unique_lock<boost::mutex> lock_(unwrapped_.future_->mutex);
          unwrapped_.future_->set_continuation_ptr(
            this->shared_from_this(), lock_);
        } else {
          this->mark_exceptional_finish_internal(
            boost::copy_exception(boost::future_uninitialized()), lock);
        }
      }
    }
  }
};

template <typename F>
struct future_unwrap_shared_state<F, void> : boost::detail::shared_state<void> {

  F wrapped_;
  typename F::value_type unwrapped_;

  explicit future_unwrap_shared_state(BOOST_THREAD_RV_REF(F) f) :
    wrapped_(boost::move(f)) {}

  void launch_continuation() {
    boost::unique_lock<boost::mutex> lock(this->mutex_);

    if (!unwrapped_.valid()) {
      if (wrapped_.has_exception()) {
        this->mark_exceptional_finish_internal(
          wrapped_.get_exception_ptr(), lock);
      } else {
        unwrapped_ = wrapped_.get();
        if (unwrapped_.valid()) {
          lock.unlock();
          boost::unique_lock<boost::mutex> lock_(unwrapped_.future_->mutex);
          unwrapped_.future_->set_continuation_ptr(
            this->shared_from_this, lock_);
        } else {
          this->mark_exceptional_finish_internal(
            boost::copy_exception(boost::future_uninitialized()), lock);
        }
      }
    } else {
      if (unwrapped_.has_exception()) {
        this->mark_exceptional_finish_internal(
          unwrapped_.get_exception_ptr(), lock);
      } else {
        this->mark_finished_with_result_internal(lock);
      }
    }
  }
};

template <typename F, typename R>
BOOST_THREAD_FUTURE<R>
  make_future_unwrap_shared_state(
    boost::unique_lock<boost::mutex>& lock, BOOST_THREAD_RV_REF(F) f) {

  boost::shared_ptr<boost::detail::future_unwrap_shared_state<F, R> > h(
    new boost::detail::future_unwrap_shared_state<F, R>(boost::move(f)));

  h->wrapped_.future_->set_continuation_ptr(h, lock);

  return BOOST_THREAD_FUTURE<R>(h);
}
} // detail

//template <typename R>
//inline BOOST_THREAD_FUTURE<R>::BOOST_THREAD_FUTURE(
//  BOOST_THREAD_RV_REF(BOOST_THREAD_FUTURE<BOOST_THREAD_FUTURE<R> >) that) :
//    base_type(that.unwrap()) {}

template <typename R>
BOOST_THREAD_FUTURE<R>
  BOOST_THREAD_FUTURE<BOOST_THREAD_FUTURE<R> >::unwrap() {
  BOOST_THREAD_ASSERT_PRECONDITION(
    this->future_.get != 0,
    boost::future_uninitialized());

  boost::shared_ptr<
    boost::detail::shared_state_base> shared_state(this->future_);
  boost::unique_lock<boost::mutex> lock(shared_state->mutex_);

  return boost::detail::make_future_unwrap_shared_state<
    BOOST_THREAD_FUTURE<BOOST_THREAD_FUTURE<
      R> >, R>(lock, boost::move(*this));
}
#endif // BOOST_THREAD_PROVIDES_FUTURE_UNWRAP
#ifdef BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY
namespace detail {

struct input_iterator_tag {};
struct vector_tag {};
struct values_tag {};
template <typename T>
struct alias_t {
  typedef T type;
};

BOOST_CONSTEXPR_OR_CONST input_iterator_tag input_iterator_tag_value = {};
BOOST_CONSTEXPR_OR_CONST vector_tag vector_tag_value = {};
BOOST_CONSTEXPR_OR_CONST values_tag values_tag_value = {};

template <typename F>
struct future_when_all_vector_shared_state :
  boost::detail::future_async_shared_state<
    void, boost::csbl::vector<F> > {
  typedef boost::csbl::vector<F> vector_type;
  typedef typename F::value_type value_type;
  vector_type v_;

  static void run(boost::shared_ptr<boost::detail::shared_state_base> that) {
    future_when_all_vector_shared_state* that_ =
      static_cast<future_when_all_vector_shared_state*>(that.get());

    try {
      boost::wait_for_all(that_->v_.begin, that_->v_.end());
      that_->mark_finished_with_result(boost::move(that_->v_));
    } catch (...) {
      that_->mark_exceptional_finish();
    }
  }

  bool run_deferred() {
    bool r = false;

    typename boost::csbl::vector<F>::iterator it = v_.begin();
    for (; it != v_.end(); ++it) {
      if (!it->run_if_is_deferred()) {
        r = true;
      }
    }
    return r;
  }

  void init() {
    if (!run_deferred()) {
      future_when_all_vector_shared_state::run(this->shared_from_this());
      return;
    }
#ifdef BOOST_THREAD_FUTURE_BLOCKING
    this->thr_ = boost::thread(
      &future_when_all_vector_shared_state::run,
      this->shared_from_this()).detach();
#else
    boost::thread(&future_when_all_vector_shared_state::run,
      this->shared_from_this()).detach();
#endif // BOOST_THREAD_FUTURE_BLOCKING
  }

  template <typename InputIter>
  future_when_all_vector_shared_state(
    input_iterator_tag, InputIter begin, InputIter end) :
      v_(std::make_move_iterator(begin), std::make_move_iterator(end)) {}

  future_when_all_vector_shared_state(
    vector_tag, BOOST_THREAD_RV_REF(boost::csbl::vector<F>) v) :
      v_(boost::move(v)) {}

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
  template <typename T, typename... Ts>
  future_when_all_vector_shared_state(
    values_tag, BOOST_THREAD_FWD_REF(T) f, BOOST_THREAD_FWD_REF(Ts) ...fs) {
    v_.push_back(boost::forward<T>(f));

    typename alias_t<char[]>::type {
      (
        v_.push_back(boost::forward<T>(fs)),
        '0'
      )...,
      '0'
    };
  }
#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATES

  ~future_when_all_vector_shared_state() {}
};

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
struct wait_for_all_fctr {
  template <typename... T>
  void operator()(T&& ...v) {
    boost::wait_for_all(boost::forward<T>(v)...);
  }
};

struct wait_for_any_fctr {
  template <typename... T>
  void operator()(T&& ...v) {
    boost::wait_for_any(boost::forward<T>(v)...);
  }
};

template <typename T, std::size_t s = boost::csbl::tuple_size<T>::value>
struct accumulate_run_if_is_deferred {
  bool operator()(T& t) {
    return (!boost::csbl::get<s - 1>(t).run_if_is_deferred()) ||
           accumulate_run_if_is_deferred<T, s - 1>()(t);
  }
};

template <typename T>
struct accumulate_run_if_is_deferred<T, 0> {
  bool operator()(T&) {
    return false;
  }
};

template <typename T, typename N, typename... Ns>
struct future_when_all_tuple_shared_state :
  boost::detail::future_async_shared_state_base<T> {
  T t_;
  typedef typename boost::detail::make_tuple_indices<
    1 + sizeof ...(Ns)>::type index;

  static void run(boost::shared_ptr<boost::detail::shared_state_base> that) {
    future_when_all_tuple_shared_state* that_ =
      static_cast<future_when_all_tuple_shared_state*>(that.get());

    try {
      that_->wait_for_all(index());
      that_->mark_finished_with_result(boost::move(that_->t_));
    } catch (...) {
      that_->mark_exceptional_finish();
    }
  }

  template <size_t... I>
  void wait_for_all(boost::detail::tuple_indices<I ...>) {
#ifdef BOOST_THREAD_PROVIDES_INVOKE
    return boost::detail::invoke<void>(wait_for_all_fctr(),
      boost::csbl::get<I>(t_)...);
#else
    return wait_for_all_fctr()(boost::csbl::get<I>(t_)...);
#endif // BOOST_THREAD_PROVIDES_INVOKE
  }

  bool run_deferred() {
    return accumulate_run_if_is_deferred<T>()(t_);
  }

  void init() {
    if (!run_deferred()) {
      future_when_all_tuple_shared_state::run(this->shared_from_this());
    }
#ifdef BOOST_THREAD_FUTURE_BLOCKING
    this->thr_ = boost::thread(&future_when_all_tuple_shared_state::run,
      this->shared_from_this());
#else
    boost::thread(&future_when_all_tuple_shared_state::run,
      this->shared_from_this()).detach();
#endif // BOOST_THREAD_FUTURE_BLOCKING
  }

  template <typename F, typename... Fs>
  future_when_all_tuple_shared_state(values_tag,
    BOOST_THREAD_FWD_REF(F) f, BOOST_THREAD_FWD_REF(Fs) ...fs) :
      t_(boost::csbl::make_tuple(
        boost::forward<F>(f), boost::forward<Fs>(fs)...)) {}

  ~future_when_all_tuple_shared_state() {}
};

template <typename T, std::size_t s = boost::csbl::tuple_size<T>::value>
struct apply_any_run_if_is_deferred_or_ready {
  bool operator()(T& t) {
    if (boost::csbl::get<s - 1>(t).run_if_is_deferred_or_ready()) {
      return true;
    }
  }
};

template <typename T>
struct apply_any_run_if_is_deferred_or_ready<T, 0> {
  bool operator()(T&) {
    return false;
  }
};
#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATES
#endif // BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY
} // detail 
} // boost
