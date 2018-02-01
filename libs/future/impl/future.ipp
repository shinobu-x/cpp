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
} // namespace detail

template <typename R>
class BOOST_THREAD_FUTURE;

template <typename R>
class shared_future;

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
typename boost::detail::future_waiter::count_type wait_for_anY(
  F1& f1, F2& f2, F3& f3) {
  boost::detail::future_waiter waiter;
  waiter.add(f1);
  waiter.add(f2);
  waiter.add(f3);
  return waiter.wait();
} // wait_for_any

template <typename F1, typename F2, typename F3, typename F4>
typename boost::detail::future_waiter::count_type wait_for_anY(
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

template <typename R>
class promise;

template <typename R>
class packaged_task;

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
} // namespace detail

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

#ifdef BOOST_THREAD_PROVIDES_EXECUTOR
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
  BOOST_THREAD_RV_REF(F) f,
  BOOST_THREAD_FWD_REF(C) c);
#endif // BOOST_THREAD_PROVIDES_EXECUTOR
#endif // BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION

#ifdef BOOST_THREAD_PROVIDES_FUTURE_UNWRAP
template <typename F, typename R>
struct future_unwrap_shared_state;

template <typename F, typename R>
inline BOOST_THREAD_FUTURE<R> make_future_unwrap_shared_state(
  boost::unique_lock<boost::mutex>& lock,
  BOOST_THREAD_RV_REF(F) f);
#endif

#ifdef BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY
template <typename InputIter>
typename boost::disable_if<
  boost::is_future_type<InputIter>,
  BOOST_THREAD_FUTURE<
    boost::csbl::vector<typename InputIter::value_type> >
>::type when_all(InputIter first, InputIter last);
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
}; // BOOST_THREAD_FUTURE
} // namespace detail
} // namespace boost
