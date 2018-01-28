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

struct shared_state_base :
  boost::enable_shared_from_this<shared_state_base> {

  typedef std::list<boost::condition_variable_any*> waiters_list;
  typedef waiters_list::iterator notify_when_ready_handle;
  typedef boost::shared_ptr<shared_state_base> continuation_ptr_type;
  typedef std::vector<continuation_ptr_type> continuations_type;

  boost::exception_ptr exception;
  bool done_;
  bool is_valid_;
  bool is_deferred_;
  bool is_constructed_;
  boost::launch policy_;
  mutable boost::mutex mutex;
  boost::condition_variable waiters;
  waiters_list external_waiters;
  boost::function<void()> callback;
  continuations_type continuations;
  boost::executor_ptr_type executor;

  virtual void launch_continuation() {}

  shared_state_base() : done_(false), is_valid_(true), is_deferred_(false),
    is_constructed_(false), policy_(boost::launch::none), continuations(),
    executor() {}

  shared_state_base(boost::exceptional_ptr const& ex) : exception(ex.ptr_),
    done_(true), is_valid_(true), is_deferred_(false),
    is_constructed_(false), policy_(boost::launch::none), continuations(),
    executor() {}

  virtual ~shared_state_base() {}

  boost::executor_ptr_type get_executor() {
    return executor;
  }

  void set_executor_policy(boost::executor_ptr_type ex) {
    set_executor();
    executor = ex;
  }

  void set_executor_policy(boost::executor_ptr_type ex,
    boost::lock_guard<boost::mutex>&) {
    set_executor();
    executor = ex;
  }

  void set_executor_policy(boost::executor_ptr_type ex,
    boost::unique_lock<boost::mutex>&) {
    set_executor();
    executor = ex;
  }

  bool valid(boost::unique_lock<boost::mutex>&) {
    return is_valid_;
  }

  bool valid() {
    boost::unique_lock<boost::mutex> lk(this->mutex);
    return valid(lk);
  }

  void validate(boost::unique_lock<boost::mutex>&) {
    is_valid_ = true;
  }

  void validate() {
    boost::unique_lock<boost::mutex> lk(this->mutex);
  }

  void invalidate(boost::unique_lock<boost::mutex>&) {
    is_valid_ = false;
  }

  void set_deferred() {
    is_deferred_ = true;
    policy_ = boost::launch::deferred;
  }

  void set_async() {
    is_deferred_ = false;
    policy_ = boost::launch::async;
  }

  void set_executor() {
    is_deferred_ = false;
    policy_ = boost::launch::executor;
  }

  void do_callback(boost::unique_lock<boost::mutex>& lock) {
    if (callback && !done_) {
      boost::function<void()> local_callback = callback;
      relocker relock(lock);
      local_callback();
    }
  }

  notify_when_ready_handle notify_when_ready(
    boost::condition_variable_any& cv) {
    boost::unique_lock<boost::mutex> lock(this->mutex);
    do_callback(lock);
    return external_waiters.insert(external_waiters.end(), &cv);
  }

  void unnotify_when_ready(notify_when_ready_handle waiter) {
    boost::lock_guard<boost::mutex> lock(this->mutex);
    external_waiters.erase(waiter);
  }

  void do_continuation(boost::unique_lock<boost::mutex>& lock) {
    if (!continuations.empty()) {
      continuations_type this_continuations = continuations;
      continuations.clear();
      relocker relock(lock);

      continuations_type::iterator it = this_continuations.begin();
      for (; it != this_continuations.end(); ++it)
        (*it)->launch_continuation();
    }
  }

  virtual void set_continuation_ptr(continuation_ptr_type continuation,
    boost::unique_lock<boost::mutex>& lock) {
    continuations.push_back(continuation);
    if (done_)
      do_continuation(lock);
  }

  void mark_finished_internal(boost::unique_lock<boost::mutex>& lock) {
    done_ = true;
    waiters.notify_all();

    waiters_list::iterator it = external_waiters.begin();
    for (; it != external_waiters.end(); ++it)
      (*it)->notify_all();

    do_continuation(lock);
  }

  void make_ready() {
    boost::unique_lock<boost::mutex> lock(this->mutex);
    mark_finished_internal(lock);
  }

  virtual void execute(boost::unique_lock<boost::mutex>&) {}

  virtual bool run_if_is_deferred() {
    boost::unique_lock<boost::mutex> lock(this->mutex);
    if (is_deferred_) {
      is_deferred_ = false;
      execute(lock);
      return true;
    } else
      return false;
  }

  virtual bool run_if_is_deferred_or_ready() {
    boost::unique_lock<boost::mutex> lock(this->mutex);
    if (is_deferred_) {
      is_deferred_ = false;
      execute(lock);
      return true;
    } else
      return done_;
  }

  void wait_internal(boost::unique_lock<boost::mutex>& lock,
    bool rethrow = true) {
    do_callback(lock);

    if (is_deferred_) {
      is_deferred_ = false;
      execute(lock);
    }

    while (!done_)
      waiters.wait(lock);

    if (rethrow && exception)
      boost::rethrow_exception(exception);
  }

  virtual void wait(boost::unique_lock<boost::mutex>& lock,
    bool rethrow = true) {
    wait_internal(lock, rethrow);
  }

  void wait(bool rethrow = true) {
    boost::unique_lock<boost::mutex> lock(this->mutex);
    wait(lock, rethrow);
  }

  bool timed_wait_until(boost::system_time const& target_time) {
    boost::unique_lock<boost::mutex> lock(this->mutex);

    if (is_deferred_)
      return false;

    do_callback(lock);

    while (!done_) {
      bool const success = waiters.timed_wait(lock, target_time);
      if (!success && !done_)
        return false;
    }

    return true;
  }

  template <class Clock, class Duration>
  boost::future_status
  wait_until(const boost::chrono::time_point<Clock, Duration>& abs_time) {
    boost::unique_lock<boost::mutex> lock(this->mutex);

    if (is_deferred_)
      return boost::future_status::deferred;

    do_callback(lock);

    while (!done_) {
      boost::cv_status const status = waiters.wait_until(lock, abs_time);
      if (status == boost::cv_status::timeout && !done_)
        return boost::future_status::timeout;
    }

    return boost::future_status::ready;
  }

  void mark_exceptional_finish_internal(
    boost::exception_ptr const& ex,
    boost::unique_lock<boost::mutex>& lock) {
    exception = ex;
    mark_finished_internal(lock);
  }

  void mark_exceptional_finish() {
    boost::unique_lock<boost::mutex> lock(this->mutex);
    mark_exceptional_finish_internal(
      boost::current_exception(), lock);
  }

  void set_exception_at_thread_exit(boost::exception_ptr e) {
    boost::unique_lock<boost::mutex> lock(this->mutex);

    if (has_value(lock))
      boost::throw_exception(boost::promise_already_satisfied());

    exception = e;
    this->is_constructed_ = true;
    boost::detail::make_ready_at_thread_exit(shared_from_this());
  }

  bool has_value() const {
    boost::lock_guard<boost::mutex> lock(this->mutex);
    return done_ && !exception;
  }

  bool has_value(boost::unique_lock<boost::mutex>&) const {
    return done_ && !exception;
  }

  bool has_exception() const {
    boost:lock_guard<boost::mutex> lock(this->mutex);
    return done_ && exception;
  }

  boost::launch launch_policy(boost::unique_lock<boost::mutex>&) const {
    return policy_;
  }

  boost::future_state::state get_state(
    boost::unique_lock<boost::mutex>&) const {

    if (!done_)
      return boost::future_state::waiting;
    else
      return boost::future_state::ready;
  }

  boost::future_state::state get_state() const {
    boost::lock_guard<boost::mutex> lock(this->mutex);

    if (!done_)
      return boost::future_state::waiting;
    else
      return boost::future_state::ready;
  }

  boost::exception_ptr get_exception_ptr() {
    boost::unique_lock<boost::mutex> lock(this->mutex);
    wait_internal(lock, false);
  }

  template <typename F, typename U>
  void set_wait_callback(F f, U* u) {
    boost::lock_guard<boost::mutex> lock(this->mutex);
    callback = boost::bind(f, boost::ref(*u));
  }

private:
  shared_state_base(shared_state_base const&);
  shared_state_base operator=(shared_state_base const&);
}; // shared_state_base

template <typename T>
struct shared_state : boost::detail::shared_state_base {
  typedef boost::optional<T> storage_type;
  typedef typename boost::conditional<
    boost::is_fundamental<T>::value,
    T,
    T const&>::type source_reference_type;
  typedef BOOST_THREAD_RV_REF(T) rvalue_source_type;
  typedef T move_dest_type;
  typedef const T& shared_future_get_result_type;
  storage_type result;

  shared_state() : result() {}
  shared_state(boost::exceptional_ptr const& ex) :
    boost::detail::shared_state_base(ex), result() {}
  ~shared_state() {}

  void mark_finished_with_result_internal(
    source_reference_type new_result,
    boost::unique_lock<boost::mutex>& lock) {
    result = new_result;
    this->mark_finished_internal(lock);
  }

  void mark_finished_with_result_internal(
    rvalue_source_type new_result,
    boost::unique_lock<boost::mutex>& lock) {
    result = boost::move(new_result);
    this->mark_finished_internal(lock);
  }

  template <class... Args>
  void mark_finished_with_result_internal(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_FWD_REF(Args)... args) {
    result.emplace(boost::forward<Args>(args)...);
    this->mark_finished_internal(lock);
  }

  void mark_finsihed_with_result(source_reference_type new_result) {
    boost::unique_lock<boost::mutex> lock(this->muetx);
    this->mark_finished_with_result_internal(new_result, lock);
  }

  void mark_finished_with_result(rvalue_source_type new_result) {
    boost::unique_lock<boost::mutex> lock(this->lock);
    mark_finished_with_result_internal(boost::move(new_result), lock);
  }

  storage_type& get_storage(boost::unique_lock<boost::mutex>& lock) {
    wait_internal(lock);
    return result;
  }

  virtual move_dest_type get(boost::unique_lock<boost::mutex>& lock) {
    return boost::move(*get_storage(lock));
  }

  move_dest_type get() {
    boost::unique_lock<boost::mutex> lock(this->lock);
    return this->get(lock);
  }

  virtual shared_future_get_result_type get_s(
    boost::unique_lock<boost::mutex>& lock) {
    return *get_storage(lock);
  }

  shared_future_get_result_type get_s() {
    boost::unique_lock<boost::mutex> lock(this->mutex);
    return this->get_s(lock);
  }

  void set_value_at_thread_exit(source_reference_type new_result) {
    boost::unique_lock<boost::mutex> lock(this->lock);

    if (this->has_value(lock))
      boost::throw_exception(boost::promise_already_satisfied());

    result = new_result;
    this->is_constructed_ = true;
    boost::detail::make_ready_at_thread_exit(shared_from_this());
  }

  void set_value_at_thread_exit(rvalue_source_type new_result) {
    boost::unique_lock<boost::mutex> lock(this->mutex);

    if (this->has_value(lock))
      boost::throw_exception(boost::promise_already_satisfied());

    result = new_result;
    this->is_constructed_ = true;
    boost::detail::make_ready_at_thread_exit(shared_from_this());
  }

private:
  shared_state(shared_state const&);
  shared_state operator=(shared_state const&);
}; // shared_state

template <typename T>
struct shared_state<T&> : boost::detail::shared_state_base {
  typedef T* storage_type;
  typedef T& source_reference_type;
  typedef T& move_dest_type;
  typedef T& shared_future_get_result_type;
  T* result;

  shared_state() : result(0) {}
  shared_state(boost::exception_ptr const& ex) :
    boost::detail::shared_state_base(ex),
    result(0) {}
  ~shared_state() {}

  void mark_finished_with_result_internal(
    source_reference_type new_result,
    boost::unique_lock<boost::mutex>& lock) {
    result = new_result;
    mark_finished_internal(lock);
  }

  void mark_finished_with_result(source_reference_type new_result) {
    boost::unique_lock<boost::mutex> lock(this->lock);
    mark_finished_with_result_internal(new_result, lock);
  }

  virtual T& get(boost::unique_lock<boost::mutex>& lock) {
    wait_internal(lock);
    return *result;
  }

  T& get() {
    boost::unique_lock<boost::mutex> lock(this->lock);
    return get(lock);
  }

  virtual T& get_s(boost::unique_lock<boost::mutex> lock) {
    wait_internal(lock);
    return *result;
  }

  T& get_s() {
    boost::unique_lock<boost::mutex> lock(this->lock);
    return get_s(lock);
  }

  void set_value_at_thread_exit(T& new_result) {
    boost::unique_lock<boost::mutex> lock(this->lock);

    if (this->has_value(lock))
      boost::throw_exception(boost::promise_already_satisfied());

    result = new_result;
    this->is_constructed_ = true;
    boost::detail::make_ready_at_thread_exit(shared_from_this());
  }

private:
  shared_state(shared_state const&);
  shared_state& operator=(shared_state const&);
}; // shared_state

template <>
struct shared_state<void> : boost::detail::shared_state_base {
  typedef void shared_future_get_result_type;
  typedef void move_dest_type;

  shared_state() {}
  shared_state(boost::exceptional_ptr const& ex) :
    boost::detail::shared_state_base(ex) {}
  ~shared_state() {}

  void mark_finished_with_result_internal(
    boost::unique_lock<boost::mutex>& lock) {
    mark_finished_internal(lock);
  }

  void mark_finished_with_result() {
    boost::unique_lock<boost::mutex> lock(this->mutex);
    mark_finished_with_result_internal(lock);
  }

  virtual void get(boost::unique_lock<boost::mutex>& lock) {
    this->wait_internal(lock);
  }

  void get() {
    boost::unique_lock<boost::mutex> lock(this->mutex);
    this->get(lock);
  }

  virtual void get_s(boost::unique_lock<boost::mutex>& lock) {
    this->wait_internal(lock);
  }

  void get_s() {
    boost::unique_lock<boost::mutex> lock(this->mutex);
    this->get_s(lock);
  }

  void set_value_at_thread_exit() {
    boost::unique_lock<boost::mutex> lock(this->mutex);

    if (this->has_value(lock))
      boost::throw_exception(boost::promise_already_satisfied());

    this->is_constructed_ = true;
    boost::detail::make_ready_at_thread_exit(shared_from_this());
  }

private:
  shared_state(shared_state const&);
  shared_state& operator=(shared_state const&);
}; // shared_state

template <typename S>
struct future_async_shared_state_base : shared_state<S> {
  typedef shared_state<S> base_type;
protected:
  boost::thread th_;

  void join() {
    if (boost::this_thread::get_id() == th_.get_id()) {
      th_.detach();
      return;
    }
    if (th_.joinable())
      th_.join();
  }

public:
  future_async_shared_state_base() {
    this->set_async();
  }

  ~future_async_shared_state_base() {
    join();
  }

  virtual void wait(boost::unique_lock<boost::mutex>& lock, bool rethrow) {
    {
      relocker relock(lock);
      join();
    }
    this->base_type::wait(lock, rethrow);
  }
}; // future_async_shared_state_base

template <typename S, typename F>
struct future_async_shared_state : future_async_shared_state_base<S> {
  future_async_shared_state() {}

  void init(BOOST_THREAD_FWD_REF(F) f) {
    this->th_ = boost::thread(
      &future_async_shared_state::run,
      boost::static_shared_from_this(this),
      boost::forward<F>(f));

    boost::thread(
      &future_async_shared_state::run,
      boost::static_shared_from_this(this),
      boost::forward<F>(f)).detach();
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
  public future_async_shared_state_base<void> {

  void init(BOOST_THREAD_FWD_REF(F) f) {
    this->th_ = boost::thread(
      &future_async_shared_state::run,
      boost::static_shared_from_this(this),
      boost::move(f));
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

template <typename S, typename F>
struct future_async_shared_state<S&, F> :
  future_async_shared_state_base<S&> {

  void init(BOOST_THREAD_FWD_REF(F) f) {
    this->th_ = boost::thread(
      &future_async_shared_state::run,
      boost::static_shared_from_this(this),
      boost::move(f));
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

template <typename S, typename F>
struct future_deferred_shared_state : shared_state<S> {
  F func_;

  explicit future_deferred_shared_state(BOOST_THREAD_FWD_REF(F) f) :
    func_(boost::move(f)) {
    this->set_deferred();
  }

  virtual void execute(boost::unique_lock<boost::mutex>& lock) {
    try {
      F local_func = boost::move(func_);
      relocker relock(lock);
      S r = local_func();
      relock.lock();
      this->mark_finished_with_result_internal(boost::move(r), lock);
    } catch (...) {
      this->mark_exceptional_finish_internal(boost::current_exception(), lock);
    }
  }
}; // future_deferred_shared_state

template <typename S, typename F>
struct future_deferred_shared_state<S&, F> : shared_state<S&> {
  F func_;

  explicit future_deferred_shared_state(BOOST_THREAD_FWD_REF(F) f) :
    func_(std::move(f)) {
    this->set_deferred();
  }

  virtual void execute(boost::unique_lock<boost::mutex>& lock) {
    try {
      this->mark_finished_with_result_internal(func_(), lock);
    } catch (...) {
      this->mark_exceptional_finish_internal(boost::current_exception(), lock);
    }
  }
}; // future_deferred_shared_state

template <typename F>
struct future_deferred_shared_state<void, F> : shared_state<void> {
  F func_;

  explicit future_deferred_shared_state(BOOST_THREAD_FWD_REF(F) f) :
    func_(boost::move(f)) {
    this->set_deferred();
  }

  virtual void execute(boost::unique_lock<boost::mutex>& lock) {
    try {
      F locak_func = boost::move(func_);
      relocker relock(lock);
      locak_func();
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
  };

  struct all_futures_lock {
    typedef std::ptrdiff_t count_type_portable;
    count_type_portable count;
    boost::scoped_array<
      boost::unique_lock<boost::mutex> > locks;

    all_futures_lock(std::vector<registered_waiter>& futures) :
      count(futures.size()),
      locks(new boost::unique_lock<boost::mutex>[count]) {
      for (count_type_portable i = 0; i < count; ++i) {
        locks[i] = BOOST_THREAD_MAKE_RV_REF(
          boost::unique_lock<boost::mutex>(futures[i].future_->mutex));
      }
    }

    void lock() {
      boost::lock(locks.get(), locks.get() + count);
    }

    void unlock() {
      for (count_type_portable i = 0; i < count; ++i) {
        if (locks[i].owns_lock()) {
          locks[i].unlock();
        }
      }
    }
  };

  boost::condition_variable_any cv;
  std::vector<registered_waiter> futures_;
  count_type future_count;

public:
  future_waiter() : future_count(0) {}

  template <typename F>
  void add(F& f) {
    if (f.future_) {
      registered_waiter waiter(f.future_, f.future_->notify_when_ready(cv),
        future_count);

      try {
        futures_.push_back(waiter);
      } catch (...) {
        f.future_->unnotify_when_ready(waiter.handle_);
        throw;
      }
      ++future_count;
    }
  }

  template <typename F1, typename... Fn>
  void add(F1& f1, Fn& ...fn) {
    add(f1);
    add(fn...);
  }

  count_type wait() {
    all_futures_lock lock(futures_);

    for (;;) {
      for (count_type i = 0; i < futures_.size(); ++i) {
        if (futures_[i].future_->done_) {
          return futures_[i].index_;
        }
      }
      cv.wait(lock);
    }
  }

  ~future_waiter() {
    for (count_type i = 0; i < futures_.size(); ++i) {
      futures_[i].future_->unnotify_when_ready(futures_[i].handle_);
    }
  }
};
} // namespace detail

template <typename R>
class BOOST_THREAD_FUTURE;

template <typename R>
class shared_future;

template <typename T>
struct is_future_type<BOOST_THREAD_FUTURE<T> > : boost::true_type {};

template <typename T>
struct is_future_type<shared_future<T> > : boost::true_type {};

template <typename F1, typename... Fn>
typename boost::enable_if<
  is_future_type<F1>,
  typename boost::detail::future_waiter::count_type
  >::type wait_for_any(F1& f1, Fn& ...fn) {
    boost::detail::future_waiter waiter;
    waiter.add(f1, fn...);
    return waiter.wait();
}

template <typename R>
class promise;

template <typename R>
class packaged_task;

namespace detail { // boost::detail

class base_future {
public:
};

template <typename R>
class basic_future : public base_future {
protected:
public:
  typedef typename boost::detail::shared_state<R> shared_state_type;
  typedef boost::shared_ptr<shared_state_type> future_ptr;
  typedef typename shared_state_type::move_dest_type move_dest_type;

  static future_ptr make_exceptional_future_ptr(
    boost::exceptional_ptr const& ex) {
    return future_ptr(new shared_state_type(ex));
  }

  future_ptr future_;

  basic_future(future_ptr future) : future_(future) {}

  typedef future_state::state state;

  BOOST_THREAD_MOVABLE_ONLY(basic_future)
  basic_future() : future_() {}

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
  }

  void swap(basic_future& that) BOOST_NOEXCEPT {
    future_.swap(that.future_);
  }

  state get_state(boost::unique_lock<boost::mutex>& lock) const {
    if (!future_)
      return future_state::uninitialized;
    return future_->get_state(lock);
  }

  state get_state() const {
    if (!future_)
      return future_state::uninitialized;
    return future_->get_state();
  }

  bool is_ready() const {
    return get_state() == future_state::ready;
  }

  bool is_ready(boost::unique_lock<boost::mutex>& lock) const {
    return get_state(lock) == future_state::ready;
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
    return future_ ? future_->get_exception_ptr() : boost::exception_ptr();
  }

  bool valid() const BOOST_NOEXCEPT {
    return future_.get() != 0 && future_->valid();
  }

  void wait() const {
    if (!future_)
      boost::throw_exception(boost::future_uninitialized());
    future_->wait(false);
  }

  typedef boost::detail::shared_state_base base_type;
  typedef base_type::notify_when_ready_handle notify_when_ready_handle;

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
    return future_->notify_when_ready(cv);
  }

  void unnotify_when_ready(notify_when_ready_handle& h) {

    if(!future_) {
      boost::throw_exception(boost::future_uninitialized());
    }
    return future_->unnotify_when_ready(h);
  }

  template <class Rep, class Period>
  future_status
  wait_for(const chrono::duration<Rep, Period>& real_time) const {
    return wait_until(chrono::steady_clock::now() + real_time);
  }

  template <class Clock, class Duration>
  future_status
  wait_until(const chrono::time_point<Clock, Duration>& abs_time) const {

    if (!future_) {
      boost::throw_exception(future_uninitialized());
    }
    return future_->wait_until(abs_time);
  }
}; // basic_future
} // boost::detail

BOOST_THREAD_DCL_MOVABLE_BEG(R)
boost::detail::basic_future<R>
BOOST_THREAD_DCL_MOVABLE_END

namespace detail {
  template <typename S, typename F>
  BOOST_THREAD_FUTURE<S> make_future_async_shared_state(
  BOOST_THREAD_FWD_REF(F) f);
  template <typename S, typename F>
  BOOST_THREAD_FUTURE<S> make_future_deferred_shared_state(
  BOOST_THREAD_FWD_REF(F) f);

  template <typename F, typename S, typename C>
  struct future_async_continuation_shared_state;
  template <typename F, typename S, typename C>
  struct future_deferred_continuation_shared_state;

  template <typename F, typename S, typename C>
  BOOST_THREAD_FUTURE<S> make_future_async_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(F) c);

  template <typename F, typename S, typename C>
  BOOST_THREAD_FUTURE<S> make_future_sync_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename S, typename C>
  BOOST_THREAD_FUTURE<S> make_future_deferred_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename S, typename C>
  BOOST_THREAD_FUTURE<S> make_shared_future_async_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    F f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename S, typename C>
  BOOST_THREAD_FUTURE<S> make_shared_future_sync_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    F f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename S, typename C>
  BOOST_THREAD_FUTURE<S> make_shared_future_deferred_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    F f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename E, typename F, typename S, typename C>
  BOOST_THREAD_FUTURE<S> make_future_executor_continuation_shared_state(
    E& ex,
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename E, typename F, typename S, typename C>
  BOOST_THREAD_FUTURE<S> make_shared_future_executor_continuation_shared_state(
    E& ex,
    boost::unique_lock<boost::mutex>& lock,
    F f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename E, typename F, typename S>
  BOOST_THREAD_FUTURE<S> make_future_executor_shared_state(
    E& ex,
    BOOST_THREAD_FWD_REF(F) f);

  template <typename F, typename S>
  struct future_unwrap_shared_state;
  template <typename F, typename S>
  inline BOOST_THREAD_FUTURE<S> make_future_unwrap_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f);
} // namespace detail

template <typename InputIter>
typename boost::disable_if<is_future_type<InputIter>,
  BOOST_THREAD_FUTURE<
    boost::csbl::vector<typename InputIter::value_type
  > > >::type when_all(InputIter first, InputIter last);
inline BOOST_THREAD_FUTURE<boost::csbl::tuple<> > when_all();

template <typename T, typename... Tn>
BOOST_THREAD_FUTURE<boost::csbl::tuple<
  typename boost::decay<T>::type,
  typename boost::decay<Tn>::type...> > when_all(
  BOOST_THREAD_FWD_REF(T) f,
  BOOST_THREAD_FWD_REF(Tn) ...futures);

template <typename InputIter>
typename boost::disable_if<is_future_type<InputIter>,
  BOOST_THREAD_FUTURE<
    boost::csbl::vector<typename InputIter::value_type
  > > >::type when_any(InputIter first, InputIter last);
inline BOOST_THREAD_FUTURE<boost::csbl::tuple<> > when_any();

template <typename T, typename... Tn>
BOOST_THREAD_FUTURE<boost::csbl::tuple<
  typename boost::decay<T>::type,
  typename boost::decay<Tn>::type...> > when_any(
  BOOST_THREAD_FWD_REF(T) f,
  BOOST_THREAD_FWD_REF(Tn) ...futures);

template <typename R>
class BOOST_THREAD_FUTURE : public boost::detail::basic_future<R> {
private:
  typedef boost::detail::basic_future<R> base_type;
  typedef typename base_type::future_ptr future_ptr;

  friend class shared_future<R>;
  friend class promise<R>;

  template <typename, typename, typename>
  friend struct future_async_continuation_shared_state;
  template <typename, typename, typename>
  friend struct future_deferred_continuation_shared_state;

  template <typename F, typename S, typename C>
  friend BOOST_THREAD_FUTURE<S> make_future_async_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename S, typename C>
  friend BOOST_THREAD_FUTURE<S> make_future_sync_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename S, typename C>
  friend BOOST_THREAD_FUTURE<S> make_future_deferred_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename S, typename C>
  friend BOOST_THREAD_FUTURE<S>
  make_shared_future_async_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    F f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename S, typename C>
  friend BOOST_THREAD_FUTURE<S>
  make_shared_future_sync_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    F f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename S, typename C>
  friend BOOST_THREAD_FUTURE<S>
  make_shared_future_deferred_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    F f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename E, typename F, typename S, typename C>
  friend BOOST_THREAD_FUTURE<S> make_future_executor_continuation_shared_state(
    E& ex,
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename E, typename F, typename S, typename C>
  friend BOOST_THREAD_FUTURE<S>
  make_shared_future_executor_continuation_shared_state(
    E& ex,
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename E, typename F, typename S>
  friend BOOST_THREAD_FUTURE<S> make_future_executor_shared_state(
    E& ex,
    BOOST_THREAD_FWD_REF(F) f);

  template <typename E, typename S, typename C>
  friend BOOST_THREAD_FUTURE<S> make_future_unwrap_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(E) f);

  template <typename F, typename S>
  friend BOOST_THREAD_FUTURE<S> make_future_unwrap_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f);

  template <typename InputIter>
  friend typename boost::enable_if<is_future_type<InputIter>,
    BOOST_THREAD_FUTURE<boost::csbl::vector<typename InputIter::value_type
    > > >::type when_all(InputIter first, InputIter last);

  template <typename T, typename... Tn>
  friend BOOST_THREAD_FUTURE<boost::csbl::tuple<
    typename boost::decay<T>::type,
    typename boost::decay<Tn>::type...> > when_all(
    BOOST_THREAD_FWD_REF(T) f,
    BOOST_THREAD_FWD_REF(Tn) ...futures);

  template <typename InputIter>
  friend typename boost::disable_if<is_future_type<InputIter>,
    BOOST_THREAD_FUTURE<boost::csbl::vector<typename InputIter::value_type
    > > >::type when_any(InputIter first, InputIter last);

  template <typename T, typename... Tn>
  friend BOOST_THREAD_FUTURE<boost::csbl::tuple<
    typename boost::decay<T>::type,
    typename boost::decay<Tn>::type... > > when_any(
    BOOST_THREAD_FWD_REF(T) f,
    BOOST_THREAD_FWD_REF(Tn) ...futures);

  template <class>
  friend class packaged_task;

  friend class boost::detail::future_waiter;

  template <class S, class F>
  friend BOOST_THREAD_FUTURE<S> make_future_async_shared_state(
    BOOST_THREAD_FWD_REF(F) f);

  template <class S, class F>
  friend BOOST_THREAD_FUTURE<S> make_future_deferred_shared_state(
    BOOST_THREAD_FWD_REF(F) f);

  typedef typename base_type::move_dest_type move_dest_type;

  BOOST_THREAD_FUTURE(future_ptr future) : base_type(future) {}

public:
  BOOST_THREAD_MOVABLE_ONLY(BOOST_THREAD_FUTURE)
  typedef boost::future_state::state state;
  typedef R value_type;

  BOOST_CONSTEXPR BOOST_THREAD_FUTURE() {}
  BOOST_THREAD_FUTURE(boost::exceptional_ptr const& ex) : base_type(ex) {}

  ~BOOST_THREAD_FUTURE() {}

  BOOST_THREAD_FUTURE(
    BOOST_THREAD_RV_REF(BOOST_THREAD_FUTURE) that) BOOST_NOEXCEPT :
    base_type(boost::move(static_cast<base_type&>(BOOST_THREAD_RV(that)))) {}
  inline BOOST_THREAD_FUTURE(
    BOOST_THREAD_RV_REF(BOOST_THREAD_FUTURE<BOOST_THREAD_FUTURE<R> >) that);

  BOOST_THREAD_FUTURE& operator=(
    BOOST_THREAD_RV_REF(BOOST_THREAD_FUTURE) that) BOOST_NOEXCEPT {
    this->base_type::operator=(
    boost::move(static_cast<base_type&>(BOOST_THREAD_RV(that))));
    return *this;
  }

  shared_future<R> share() {
    return shared_future<R>(boost::move(*this));
  }

  void swap(BOOST_THREAD_FUTURE& that) {
    static_cast<base_type*>(this)->swap(that);
  }

  void set_async() {
    this->future_->set_async();
  }

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
    if (this->future_ == 0) {
      boost::throw_exception(boost::future_uninitialized());
    }

    boost::unique_lock<boost::mutex> lock(this->future_->mutex);
    if (!this->future_->valid(lock)) {
      boost::throw_exception(boost::future_uninitialized());
    }
    this->future_->invalidate(lock);

    return this->future_->get(lock);
  }

  template <typename R2>
  typename boost::disable_if<
    boost::is_void<R2>,
    move_dest_type>::type get_or(BOOST_THREAD_RV_REF(R2) v) {
    if (this->future_ == 0) {
      boost::throw_exception(boost::future_uninitialized());
    }

    boost::unique_lock<boost::mutex> lock(this->future_->mutex);

    if (!this->future_->valid(lock)) {
      boost::throw_exception(boost::future_uninitialized());
    }

    this->future_->wait(lock, false);
    this->future_->invalidate(lock);

    if (this->future_->has_value(lock)) {
      return this->future_->get(lock);
    } else {
      return boost::move(v);
    }
  }

  template <typename R2>
  typename boost::disable_if<
    boost::is_void<R2>,
    move_dest_type>::type get_or(R2 const& v) {
    if (this->future_ == 0) {
      boost::throw_exception(boost::future_uninitialized());
    }

    boost::unique_lock<boost::mutex> lock(this->future_->mutex);

    if (!this->future_->valid(lock)) {
      boost::throw_exception(boost::future_uninitialized());
    }

    this->future_->wait(lock, false);
    this->future_->invalidate(lock);

    if (this->future_->has_value(lock)) {
      return this->future_->get(lock);
    } else {
      return v;
    }
  }

  template <typename F>
  inline BOOST_THREAD_FUTURE<
    typename boost::result_of<
      F(BOOST_THREAD_FUTURE)>::type> then(
        BOOST_THREAD_FWD_REF(F) func);

  template <typename F>
  inline BOOST_THREAD_FUTURE<
    typename boost::result_of<
      F(BOOST_THREAD_FUTURE)>::type> then(
        boost::launch policy, BOOST_THREAD_FWD_REF(F) func);

  template <typename E, typename F>
  inline BOOST_THREAD_FUTURE<
    typename boost::result_of<
      F(BOOST_THREAD_FUTURE)>::type> then(
        E& ex, BOOST_THREAD_FWD_REF(F) func);

  template <typename R2>
  inline typename boost::disable_if<
    boost::is_void<R2>,
    BOOST_THREAD_FUTURE<R> >::type fallback_to(BOOST_THREAD_RV_REF(R2) v);

  template <typename R2>
  inline typename boost::disable_if<
    boost::is_void<R2>,
    BOOST_THREAD_FUTURE<R2> >::type fallback_to(R2 const& v);
}; // BOOST_THREAD_FUTURE

BOOST_THREAD_DCL_MOVABLE_BEG(T)
BOOST_THREAD_FUTURE<T>
BOOST_THREAD_DCL_MOVABLE_END

template <typename R2>
class BOOST_THREAD_FUTURE<BOOST_THREAD_FUTURE<R2> > :
  public boost::detail::basic_future<BOOST_THREAD_FUTURE<R2> > {
  typedef BOOST_THREAD_FUTURE<R2> R;
private:
  typedef boost::detail::basic_future<R> base_type;
  typedef typename base_type::future_ptr future_ptr;

  friend class shared_future<R>;
  friend class promise<R>;

  template <typename, typename, typename>
  friend struct boost::detail::future_async_continuation_shared_state;
  template <typename, typename, typename>
  friend struct boost::detail::future_deferred_continuation_shared_state;

  template <typename F, typename S, typename C>
  friend BOOST_THREAD_FUTURE<S>
  boost::detail::make_future_async_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename S, typename C>
  friend BOOST_THREAD_FUTURE<S>
  boost::detail::make_future_sync_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename S, typename C>
  friend BOOST_THREAD_FUTURE<S>
  boost::detail::make_future_deferred_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename S, typename C>
  friend BOOST_THREAD_FUTURE<S>
  boost::detail::make_shared_future_async_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    F f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename S, typename C>
  friend BOOST_THREAD_FUTURE<S>
  boost::detail::make_shared_future_sync_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    F f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename S, typename C>
  friend BOOST_THREAD_FUTURE<S>
  boost::detail::make_shared_future_deferred_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    F f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename E, typename F, typename S, typename C>
  friend BOOST_THREAD_FUTURE<S>
  boost::detail::make_future_executor_continuation_shared_state(
    E& ex,
    boost::unique_lock<boost::mutex>& lock,
    F f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename E, typename F, typename S, typename C>
  friend BOOST_THREAD_FUTURE<S>
  boost::detail::make_shared_future_executor_continuation_shared_state(
    E& ex,
    boost::unique_lock<boost::mutex>& lock,
    F f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename E, typename F, typename S>
  friend BOOST_THREAD_FUTURE<S>
  boost::detail::make_future_executor_shared_state(
    E& ex,
    BOOST_THREAD_FWD_REF(F) f); 

  template <typename F, typename S>
  friend struct boost::detail::future_unwrap_shared_state;

  template <typename F, typename S>
  friend BOOST_THREAD_FUTURE<S>
  boost::detail::make_future_unwrap_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f);

  template <typename InputIter>
  friend typename boost::disable_if<
    is_future_type<InputIter>,
    BOOST_THREAD_FUTURE<boost::csbl::vector<
      typename InputIter::value_type> > >::type when_all(
    InputIter first, InputIter last);

  friend inline BOOST_THREAD_FUTURE<boost::csbl::tuple<> > when_all();

  template <typename T, typename... Tn>
  friend BOOST_THREAD_FUTURE<boost::csbl::tuple<
    typename boost::decay<T>::type,
    typename boost::decay<Tn>::type...> > when_all(
    BOOST_THREAD_FWD_REF(T) f,
    BOOST_THREAD_FWD_REF(Tn) ...futures);

  template <typename InputIter>
  friend typename boost::disable_if<
    is_future_type<InputIter>,
    BOOST_THREAD_FUTURE<boost::csbl::vector<
      typename InputIter::value_type> > >::type when_any(
    InputIter first, InputIter last);

  friend inline BOOST_THREAD_FUTURE<boost::csbl::tuple<> > when_any();

  template <typename T, typename... Tn>
  friend BOOST_THREAD_FUTURE<boost::csbl::tuple<
    typename boost::decay<T>::type,
    typename boost::decay<Tn>::type...> > when_any(
    BOOST_THREAD_FWD_REF(T) f,
    BOOST_THREAD_FWD_REF(Tn) ...futures);

  template <typename>
  friend class packaged_task;
  friend class boost::detail::future_waiter;

  template <typename F, typename S>
  friend BOOST_THREAD_FUTURE<S>
  boost::detail::make_future_async_shared_state(
    BOOST_THREAD_FWD_REF(F) f);

  template <typename F, typename S>
  friend BOOST_THREAD_FUTURE<S>
  boost::detail::make_future_deferred_shared_state(
    BOOST_THREAD_FWD_REF(F) f);

  typedef typename base_type::move_dest_type move_dest_type;

  BOOST_THREAD_FUTURE(future_ptr future) : base_type(future) {}

public:
  BOOST_THREAD_MOVABLE_ONLY(BOOST_THREAD_FUTURE)
  typedef future_state::state state;
  typedef R value_type;

  BOOST_CONSTEXPR BOOST_THREAD_FUTURE() {}

  BOOST_THREAD_FUTURE(boost::exception_ptr const& ex) : base_type(ex) {}

  ~BOOST_THREAD_FUTURE() {}

  BOOST_THREAD_FUTURE(
    BOOST_THREAD_RV_REF(BOOST_THREAD_FUTURE) other) BOOST_NOEXCEPT :
    base_type(boost::move(static_cast<base_type&>(BOOST_THREAD_RV(other)))) {}

  inline BOOST_THREAD_FUTURE(BOOST_THREAD_RV_REF(
    BOOST_THREAD_FUTURE<BOOST_THREAD_FUTURE<R> >) other);

  explicit BOOST_THREAD_FUTURE(BOOST_THREAD_RV_REF(shared_future<R>) other) :
    base_type(boost::move(static_cast<base_type&>(BOOST_THREAD_RV(other)))) {}

  BOOST_THREAD_FUTURE& operator=(BOOST_THREAD_RV_REF(
    BOOST_THREAD_FUTURE) other) BOOST_NOEXCEPT {
    this->base_type::operator=(boost::move(static_cast<base_type&>(
      BOOST_THREAD_RV(other))));
    return *this;
  }

  shared_future<R> share() {
    return shared_future<R>(boost::move(*this));
  }

  void swap(BOOST_THREAD_FUTURE& that) {
    static_cast<base_type*>(this)->swap(that);
  }

  void set_async() {
    this->future_->set_async();
  }

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
    if (this->future_ == 0) {
      boost::throw_exception(boost::future_uninitialized());
    }

    boost::unique_lock<boost::mutex> lock(this->future_->mutex);
    if (!this->future_->valid(lock)) {
      boost::throw_exception(boost::future_uninitialized());
    }
    this->future_->invalidate(lock);

    return this->future_->get(lock);
  }

  move_dest_type get_or(BOOST_THREAD_RV_REF(R) v) {
    if (this->future_ == 0) {
      boost::throw_exception(boost::future_uninitialized());
    }

    boost::unique_lock<boost::mutex> lock(this->future_->mutex);

    if (!this->future_->valid(lock)) {
      boost::throw_exception(boost::future_uninitialized());
    }

    this->future_->wait(lock, false);
    this->future_->invalidate(lock);

    if (this->future_->has_value(lock)) {
      return this->future_->get(lock);
    }
  }

  template <typename F>
  inline BOOST_THREAD_FUTURE<typename boost::result_of<
    F(BOOST_THREAD_FUTURE)>::type> then(BOOST_THREAD_FWD_REF(F) func);

  template <typename F>
  inline BOOST_THREAD_FUTURE<typename boost::result_of<
    F(BOOST_THREAD_FUTURE)>::type> then(
      launch policy, BOOST_THREAD_FWD_REF(F) func);

  template <typename E, typename F>
  inline BOOST_THREAD_FUTURE<typename boost::result_of<
    F(BOOST_THREAD_FUTURE)>::type> then(E& ex, BOOST_THREAD_FWD_REF(F) func);

  inline BOOST_THREAD_FUTURE<R2> unwrap();

}; // BOOST_THREAD_FUTURE

template <typename R>
class shared_future : public boost::detail::basic_future<R> {
  typedef detail::basic_future<R> base_type;
  typedef typename base_type::future_ptr future_ptr;

  friend class boost::detail::future_waiter;
  friend class promise<R>;

  template <typename, typename, typename>
  friend struct boost::detail::future_async_continuation_shared_state;
  template <typename, typename, typename>
  friend struct boost::detail::future_deferred_continuation_shared_state;

  template <typename F, typename S, typename C>
  friend BOOST_THREAD_FUTURE<S>
  boost::detail::make_future_async_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename S, typename C>
  friend BOOST_THREAD_FUTURE<S>
  boost::detail::make_future_sync_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename S, typename C>
  friend BOOST_THREAD_FUTURE<S>
  boost::detail::make_future_deferred_continuation_shared_state(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c);


  template <class>
  friend class packaged_task;
  friend class packaged_task<R>;

  shared_future(future_ptr future) : base_type(future) {}

public:
  BOOST_THREAD_COPYABLE_AND_MOVABLE(shared_future)
  typedef R value_type;

  shared_future(shared_future const& that) : base_type(that.future_) {}

  typedef future_state::state state;

  BOOST_CONSTEXPR shared_future() {}

  shared_future(boost::exception_ptr const& ex) : base_type(ex) {}
  ~shared_future() {}

  shared_future& operator=(BOOST_THREAD_COPY_ASSIGN_REF(shared_future) that) {
    this->future_ = that.future_;
    return *this;
  }

  shared_future(BOOST_THREAD_RV_REF(shared_future) that) BOOST_NOEXCEPT :
    base_type(boost::move(static_cast<base_type&>(BOOST_THREAD_RV(that)))) {}

  shared_future(
    BOOST_THREAD_RV_REF(BOOST_THREAD_FUTURE<R>) that) BOOST_NOEXCEPT :
    base_type(boost::move(static_cast<base_type&>(BOOST_THREAD_RV(that)))) {}

  shared_future& operator=(
    BOOST_THREAD_RV_REF(shared_future) that) BOOST_NOEXCEPT {
    base_type::operator=(
      boost::move(static_cast<base_type&>(BOOST_THREAD_RV(that))));
    return *this;
  }

  shared_future& operator=(
    BOOST_THREAD_RV_REF(BOOST_THREAD_FUTURE<R>) that) BOOST_NOEXCEPT {
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

  typedef boost::detail::shared_state<R> shared_state;
  typename shared_state::shared_future_get_result_type get() const {
    if (!this->future_) {
      boost::throw_exception(boost::future_uninitialized());
    }
    return this->future_->get_s();
  }

  template <typename R2>
  typename boost::disable_if<
    boost::is_void<R2>,
    typename shared_state::shared_future_get_result_type>::type get_or(
    BOOST_THREAD_RV_REF(R2) v) const {
    if (!this->future_) {
      boost::throw_exception(boost::future_uninitialized());
    }

    this->future_->wait();

    if (this->future_->has_value()) {
      return this->future_->get_s();
    } else {
      return boost::move(v);
    }
  }

  template <typename F>
  inline BOOST_THREAD_FUTURE<typename boost::result_of<
    F(shared_future)>::type> then(BOOST_THREAD_FWD_REF(F) func) const;

  template <typename F>
  inline BOOST_THREAD_FUTURE<typename boost::result_of<
    F(shared_future)>::type> then(
      launch policy,
      BOOST_THREAD_FWD_REF(F) func) const;

  template <typename E, typename F>
  inline BOOST_THREAD_FUTURE<typename boost::result_of<
    F(shared_future)>::type> then(
      E& ex,
      BOOST_THREAD_FWD_REF(F) func) const;
}; // shared_future

BOOST_THREAD_DCL_MOVABLE_BEG(T)
shared_future<T>
BOOST_THREAD_DCL_MOVABLE_END

template <typename R>
class promise {
  typedef boost::detail::shared_state<R>
    shared_state;
  typedef boost::shared_ptr<shared_state>
    future_ptr;
  typedef typename shared_state::source_reference_type
    source_reference_type;
  typedef typename shared_state::rvalue_source_type
    rvalue_source_type;
  typedef typename shared_state::move_dest_type
    move_dest_type;
  typedef typename shared_state::shared_future_get_result_type
    shared_future_get_result_type;

  future_ptr future_;
  bool future_obtained;

  void lazy_init() {
#include <boost/detail/atomic_undef_macros.hpp>
    if (!boost::atomic_load(&future_)) {
      future_ptr blank;
      boost::atomic_compare_exchange(
        &future_, &blank, future_ptr(new boost::detail::shared_state<R>));
    }
#include <boost/detail/atomic_redef_macros.hpp>
  }

public:
  BOOST_THREAD_MOVABLE_ONLY(promise)

  template <class Allocator>
  promise(boost::allocator_arg_t, Allocator a) {
    typedef typename Allocator::template rebind<shared_state>::other A2;
    A2 a2(a);
    typedef boost::thread_detail::allocator_destructor<A2> D;

    future_ = future_ptr(::new(a2.allocate(1)) shared_state(), D(a2, 1));
    future_obtained = false;
  }

  promise() : future_(), future_obtained(false) {}
  ~promise() {
    if (future_) {
      boost::unique_lock<boost::mutex> lock(future_->mutex);

      if (!future_->done_ && !future_->is_constructed_) {
        future_->mark_exceptional_finish_internal(
          boost::copy_exception(broken_promise()), lock);
      }
    }
  }

  // copy
  promise(BOOST_THREAD_RV_REF(promise) rhs) BOOST_NOEXCEPT :
    future_(BOOST_THREAD_RV(rhs).future_),
    future_obtained(BOOST_THREAD_RV(rhs).future_obtained) {
    BOOST_THREAD_RV(rhs).future_.reset();
    BOOST_THREAD_RV(rhs).future_obtained = false;
  }

  promise& operator=(BOOST_THREAD_RV_REF(promise) rhs) BOOST_NOEXCEPT {
    future_ = BOOST_THREAD_RV(rhs).future_;
    future_obtained = BOOST_THREAD_RV(rhs).future_obtained;
    BOOST_THREAD_RV(rhs).future_.reset();
    BOOST_THREAD_RV(rhs).future_obtained = false;
    return *this;
  }

  void swap(promise& that) {
    future_.swap(that.future_);
    std::swap(future_obtained, that.future_obtained);
  }

  void set_executor(boost::executor_ptr_type ex) {
    lazy_init();

    if (future_.get() == 0) {
      boost::throw_exception(boost::promise_moved());
    }

    boost::lock_guard<boost::mutex> lock(future_->mutex);
    future_->set_executor_policy(ex, lock);
  }

  BOOST_THREAD_FUTURE<R> get_future() {
    lazy_init();

    if (future_.get() == 0) {
      boost::throw_exception(boost::promise_moved());
    }

    if (future_obtained) {
      boost::throw_exception(boost::future_already_retrieved());
    }

    future_obtained = true;

    return BOOST_THREAD_FUTURE<R>(future_);
  }

  void set_value(source_reference_type r) {
    lazy_init();

    boost::unique_lock<boost::mutex> lock(future_->mutex);
    if (future_->done_) {
      boost::throw_exception(boost::promise_already_satisfied());
    }
    future_->mark_finished_with_result_internal(r, lock);
  }

  void set_value(rvalue_source_type r) {
    lazy_init();

    boost::unique_lock<boost::mutex> lock(future_->mutex);
    if (future_->done_) {
      boost::throw_exception(boost::promise_already_satisfied());
    }
    future_->mark_finish_with_result_internal(boost::move(r), lock);
  }

  template <typename... Args>
  void emplace(BOOST_THREAD_FWD_REF(Args) ...args) {
    lazy_init();

    boost::unique_lock<boost::mutex> lock(future_->mutex);
    if (future_->done_) {
      boost::throw_exception(boost::promise_already_satisfied());
    }
    future_->mark_finished_with_result_internal(
      lock, boost::forward<Args>(args)...);
  }

  void set_exception(boost::exception_ptr p) {
    lazy_init();

    boost::unique_lock<boost::mutex> lock(future_->mutex);
    if (future_->done_) {
      boost::throw_exception(boost::promise_already_satisfied());
    }
    future_->mark_exceptional_finish_internal(p, lock);
  }

  template <typename Ex>
  void set_exception(Ex e) {
    set_exception(boost::copy_exception(e)); 
  }

  void set_value_at_thread_exit(source_reference_type r) {
    if (future_.get() == 0) {
      boost::throw_exception(boost::promise_moved());
    }
    future_->set_value_at_thread_exit(r);
  }

  void set_value_at_thread_exit(BOOST_THREAD_RV_REF(R) r) {
    if (future_.get() == 0) {
      boost::throw_exception(boost::promise_moved());
    }
    future_->set_value_at_thread_exit(boost::move(r));
  }

  void set_exception_at_thread_exit(boost::exception_ptr e) {
    if (future_.get() == 0) {
      boost::throw_exception(boost::promise_moved());
    }
    future_->set_exception_at_thread_exit(e);
  }

  template <typename Ex>
  void set_exception_at_thread_exit(Ex e) {
    set_exception_at_thread_exit(boost::copy_exception(e));
  }

  template <typename F>
  void set_wait_callback(F f) {
    lazy_init();
    future_->set_wait_callback(f, this);
  }
}; // promise

template <>
class promise<void> {
  typedef boost::shared_ptr<boost::detail::shared_state<void> > future_ptr;
  future_ptr future_;
  bool future_obtained;

  void lazy_init() {
    if (!boost::atomic_load(&future_)) {
      future_ptr blank;
      boost::atomic_compare_exchange(
        &future_, &blank, future_ptr(
          new boost::detail::shared_state<void>));
    }
  }
public:
  BOOST_THREAD_MOVABLE_ONLY(promise);

  template <typename Allocator>
  promise(boost::allocator_arg_t, Allocator a) {
    typedef typename Allocator::template rebind<
      boost::detail::shared_state<void> >::other A2;
    A2 a2(a);
    typedef boost::thread_detail::allocator_destructor<A2> D;

    future_ = future_ptr(
      new(a2.allocate(1)) boost::detail::shared_state<void>(), D(a2, 1));
    future_obtained = false;
  }

  promise() : future_(), future_obtained(false) {}

  ~promise() {
    if (future_) {
      boost::unique_lock<boost::mutex> lock(future_->mutex);

      if (!future_->done_ && !future_->is_constructed_) {
        future_->mark_exceptional_finish_internal(
          boost::copy_exception(broken_promise()), lock);
      }
    }
  }

  promise(BOOST_THREAD_RV_REF(promise) rhs) BOOST_NOEXCEPT :
    future_(BOOST_THREAD_RV(rhs).future_),
    future_obtained(BOOST_THREAD_RV(rhs).future_obtained) {
    BOOST_THREAD_RV(rhs).future_.reset();
    BOOST_THREAD_RV(rhs).future_obtained = false;
  }

  promise& operator=(BOOST_THREAD_RV_REF(promise) rhs) BOOST_NOEXCEPT {
    future_ = BOOST_THREAD_RV(rhs).future_;
    future_obtained = BOOST_THREAD_RV(rhs).future_obtained;
    BOOST_THREAD_RV(rhs).future_.reset();
    BOOST_THREAD_RV(rhs).future_obtained = false;
    return *this;
  }

  void swap(promise& that) {
    future_.swap(that.future_);
    std::swap(future_obtained, that.future_obtained);
  }

  BOOST_THREAD_FUTURE<void> get_future() {
    lazy_init();

    if (future_.get() == 0) {
      boost::throw_exception(promise_moved());
    }

    if (future_obtained) {
      boost::throw_exception(future_already_retrieved());
    }

    return BOOST_THREAD_FUTURE<void>(future_);
  }

  void set_value() {
    lazy_init();
    boost::unique_lock<boost::mutex> lock(future_->mutex);

    if (future_->done_) {
      boost::throw_exception(promise_already_satisfied());
    }

    future_->mark_finished_with_result_internal(lock);
  }

  void set_exception(boost::exception_ptr p) {
    lazy_init();
    boost::unique_lock<boost::mutex> lock(future_->mutex);

    if (future_->done_) {
      boost::throw_exception(promise_already_satisfied());
    }
    future_->mark_exceptional_finish_internal(p, lock);
  }

  template <typename Ex>
  void set_exception(Ex e) {
    set_exception(boost::copy_exception(e));
  }

  void set_value_at_thread_exit() {
    if (future_.get() == 0) {
      boost::throw_exception(promise_moved());
    }
    future_->set_value_at_thread_exit();
  }

  void set_exception_at_thread_exit(boost::exception_ptr e) {
    if (future_.get() == 0) {
      boost::throw_exception(promise_moved());
    }
    future_->set_exception_at_thread_exit(e);
  }

  template <typename C>
  void set_wait_callback(C c) {
    lazy_init();
    future_->set_wait_callback(c, this);
  }
}; // promise
} // namespace boost

namespace boost {
namespace container {

template <typename R, typename Allocator>
struct uses_allocator<boost::promise<R>, Allocator> : true_type {};

} // namespace container
} // namespace boost

namespace boost {
BOOST_THREAD_DCL_MOVABLE_BEG(T)
boost::promise<T>
BOOST_THREAD_DCL_MOVABLE_END
namespace detail {

template <typename R>
struct task_base_shared_state;

template <typename R, typename... Args>
struct task_base_shared_state<R(Args ...)> : shared_state<R> {
  bool started;

  task_base_shared_state() : started(false) {}

  void reset() {
    started = false;
    this->validate();
  }

  virtual void do_run(BOOST_THREAD_RV_REF(Args) ...args) = 0;
  void run(BOOST_THREAD_RV_REF(Args) ...args) {
    {
      boost::lock_guard<boost::mutex> lock(this->mutex);
      if (started) {
        boost::throw_exception(task_already_started());
      }
      started = true;
    }
    do_run(boost::move(args)...);
  }

  virtual void do_apply(BOOST_THREAD_RV_REF(Args) ...args) = 0;
  void apply(BOOST_THREAD_RV_REF(Args) ...args) {
    {
      boost::lock_guard<boost::mutex> lock(this->mutex);
      if (started) {
        boost::throw_exception(task_already_started());
      }
      started = true;
    }
    do_apply(boost::move(args)...);
  }

  void owner_destroyed() {
    boost::unique_lock<boost::mutex> lock(this->mutex);
    if (!started) {
      started = true;
      this->mark_exceptional_finish_internal(
        boost::copy_exception(boost::broken_promise()), lock);
    }
  }
}; // shared_state

template <typename F, typename R>
struct task_shared_state;

template <typename F, typename R, typename... Args>
struct task_shared_state<F, R(Args ...)> :
  task_base_shared_state<R(Args ...)> {
private:
  task_shared_state(task_shared_state&);
public:
  F f_;
  task_shared_state(F const& f) : f_(f) {}

  task_shared_state(BOOST_THREAD_RV_REF(F) f) : f_(boost::move(f)) {}

  F callable() {
    return boost::move(f_);
  }

  void do_apply(BOOST_THREAD_RV_REF(Args) ...args) {
    try {
      this->set_value_at_thread_exit(f_(boost::move(args)...));
    } catch (...) {
      this->set_exception_at_thread_exit(boost::current_exception());
    }
  }

  void do_run(BOOST_THREAD_RV_REF(Args) ...args) {
    try {
      this->mark_finish_with_result(f_(boost::move(args)...));
    } catch (...) {
      this->mark_exceptional_finish();
    }
  }
}; // task_shared_state

template <typename F, typename R, typename... Args>
struct task_shared_state<F, R&(Args...)> :
  task_base_shared_state<R&(Args...)> {
private:
  task_shared_state(task_shared_state&);
public:
  F f_;
  task_shared_state(F const& f) : f_(f) {}

  task_shared_state(BOOST_THREAD_RV_REF(F) f) : f_(boost::move(f)) {}

  F callable() {
    return f_;
  }

  void do_apply(BOOST_THREAD_RV_REF(Args) ...args) {
    try {
      this->set_value_at_thread_exit(f_(boost::move(args)...));
    } catch (...) {
      this->set_exception_at_thread_exit(boost::current_exception());
    }
  }

  void do_run(BOOST_THREAD_RV_REF(Args) ...args) {
    try {
      this->mark_finished_with_result(f_(boost::move(args)...));
    } catch (...) {
      this->mark_exceptional_finish();
    }
  }
}; // task_shared_state

template <typename R, typename... Args>
struct task_shared_state<R (*)(Args...), R(Args...)> :
  task_base_shared_state<R(Args...)> {
private:
  task_shared_state(task_shared_state&);
  typedef R (*CallableType)(Args...);
public:
  CallableType f_;
  task_shared_state(CallableType f) : f_(f) {}

  CallableType callable() {
    return f_;
  }

  void do_apply(BOOST_THREAD_RV_REF(Args) ...args) {
    try {
      this->set_value_at_thread_exit(f_(boost::move(args)...));
    } catch (...) {
      this->set_exception_at_thread_exit(boost::current_exception());
    }
  }

  void do_run(BOOST_THREAD_RV_REF(Args) ...args) {
    try {
      this->mark_finished_with_result(f_(boost::move(args)...));
    } catch (...) {
      this->mark_exceptional_finish();
    }
  }
}; // task_shared_state

template <typename R, typename... Args>
struct task_shared_state<R& (*)(Args...), R&(Args...)> :
  task_base_shared_state<R&(Args...)> {
private:
  task_shared_state(task_shared_state&);
public:
  typedef R& (*CallableType)(BOOST_THREAD_RV_REF(Args)...);
  CallableType f_;
  task_shared_state(CallableType f) : f_(f) {}

  CallableType callable() {
    return boost::move(f_);
  }

  void do_apply(BOOST_THREAD_RV_REF(Args) ...args) {
    try {
      this->set_value_at_thread_exit(f_(boost::move(args)...));
    } catch (...) {
      this->set_exception_at_thread_exit(boost::current_exception());
    }
  }

  void do_run(BOOST_THREAD_RV_REF(Args) ...args) {
    try {
      this->mark_finished_with_result(f_(boost::move(args)...));
    } catch (...) {
      this->mark_exceptional_finish();
    }
  }
}; // task_shared_state

template <typename F, typename... Args>
struct task_shared_state<F, void(Args...)> :
  task_base_shared_state<void(Args...)> {
private:
  task_shared_state(task_shared_state&);
private:
  typedef F CallableType;
  F f_;

  task_shared_state(F const& f) : f_(f) {}

  task_shared_state(BOOST_THREAD_RV_REF(F) f) : f_(boost::move(f)) {}

  F callable() {
    return boost::move(f_);
  }

  void do_apply(BOOST_THREAD_RV_REF(Args) ...args) {
    try {
      f(boost::move(args)...);
    } catch (...) {
      this->set_exception_at_thread_exit(boost::current_exception());
    }
  }

  void do_run() {
    try {
      f_();
    } catch (...) {
      this->mark_exceptional_finsih();
    }
  }
}; // task_shared_state

template <typename... Args>
struct task_shared_state<void(*)(Args...), void(Args...)> :
  task_base_shared_state<void(Args...)> {
private:
  task_shared_state(task_shared_state&);
  typedef void (*CallableType)(Args...);
public:
  CallableType f_;
  task_shared_state(CallableType f) : f_(f) {}

  CallableType callable() {
    return f_;
  }

  void do_apply(BOOST_THREAD_RV_REF(Args) ...args) {
    try {
      f_(boost::move(args)...);
    } catch (...) {
      this->set_exception_at_thread_exit(boost::current_exception());
    }
  }

  void do_run(BOOST_THREAD_RV_REF(Args) ...args) {
    try {
      f_(boost::move(args)...);
    } catch (...) {
      this->mark_exceptional_finish();
    }
  }
}; // task_shared_state
} // namespace detail

template <typename R, typename... Args>
class packaged_task<R(Args...)> {
  typedef boost::detail::task_base_shared_state<R(Args...)> task_state;
  typedef boost::shared_ptr<task_state> task_ptr;
  boost::shared_ptr<task_state> task;

  bool future_obtained;
  struct dummy;

public:
  typedef R result_type;
  BOOST_THREAD_MOVABLE_ONLY(packaged_task);

  packaged_task() : future_obtained(false) {}

  explicit packaged_task(R(*f)(), BOOST_THREAD_FWD_REF(Args) ...args) {
    typedef R(*FR)(BOOST_THREAD_FWD_REF(Args)...);
    typedef boost::detail::task_shared_state<FR, R(Args...)>
      task_shared_state_type;

    task = task_ptr(new task_shared_state_type(f, boost::move(args)...));
    future_obtained = false;
  }

  template <typename F>
  explicit packaged_task(BOOST_THREAD_FWD_REF(F) f,
    typename boost::disable_if<
      boost::is_same<typename boost::decay<F>::type,
      packaged_task>, dummy*>::type = 0) {
    typedef typename boost::decay<F>::type FR;
    typedef boost::detail::task_shared_state<FR, R(Args...)>
      task_shared_state_type;

    task = task_ptr(new task_shared_state_type(boost::forward<F>(f)));
    future_obtained = false;
  }

  template <typename Allocator>
  packaged_task(boost::allocator_arg_t, Allocator a, R(*f)()) {
    typedef R(*FR)();
    typedef boost::detail::task_shared_state<FR, R(Args...)>
      task_shared_state_type;
    typedef typename Allocator::template rebind<
      task_shared_state_type>::other A2;
    A2 a2(a);

    typedef boost::thread_detail::allocator_destructor<A2> D;

     task = task_ptr(new(a2.allocate(1)) task_shared_state_type(f), D(a2, 1));
     future_obtained = false;
  }

  template <typename F, typename Allocator>
  packaged_task(boost::allocator_arg_t, Allocator a,
    BOOST_THREAD_FWD_REF(F) f) {
    typedef typename boost::decay<F>::type FR;
    typedef boost::detail::task_shared_state<FR, R(Args...)>
      task_shared_state_type;
    typedef typename Allocator::template rebind<
      task_shared_state_type>::other A2;
    A2 a2(a);

    typedef boost::thread_detail::allocator_destructor<A2> D;

    task = task_ptr(new(a2.allocate(1)) task_shared_state_type(
      boost::forward<F>(f)), D(a2, 1));
    future_obtained = false;
  }

  ~packaged_task() {
    if (task) {
      task->owner_destroyed();
    }
  }

  packaged_task(BOOST_THREAD_RV_REF(packaged_task) that) BOOST_NOEXCEPT :
    future_obtained(BOOST_THREAD_RV(that).future_obtained) {
    task.swap(BOOST_THREAD_RV(that).task);
    BOOST_THREAD_RV(that).future_obtained = false;
  }

  packaged_task& operator=(
    BOOST_THREAD_RV_REF(packaged_task) that) BOOST_NOEXCEPT {
    packaged_task temp(boost::move(that));
    swap(temp);
    return *this;
  }

  void set_executor(boost::executor_ptr_type ex) {
    if (!valid()) {
      boost::throw_exception(task_moved());
    }
    boost::lock_guard<boost::mutex> lock(task->mutex);
    task->set_executor_policy(ex, lock);
  }

  void reset() {
    if (!valid()) {
      boost::throw_exception(future_error(
        boost::system::make_error_code(future_errc::no_state)));
    }
    task->reset();
    future_obtained = false;
  }

  void swap(packaged_task& that) BOOST_NOEXCEPT {
    task.swap(that.task);
    std::swap(future_obtained, that.future_obtained);
  }

  bool valid() const BOOST_NOEXCEPT {
    return task.get() != 0;
  }

  BOOST_THREAD_FUTURE<R> get_future() {
    if (!task) {
      boost::throw_exception(task_moved());
    } else if (!future_obtained) {
      future_obtained = true;
      return BOOST_THREAD_FUTURE<R>(task);
    } else {
      boost::throw_exception(future_already_retrieved());
    }
  }

  void operator()(Args ...args) {
    if (!task) {
      boost::throw_exception(task_moved());
    }
    task->run(boost::move(args)...);
  }

  void make_ready_at_thread_exit(Args ...args) {
    if (!task) {
      boost::throw_exception(task_moved());
    }

    if (task->has_value) {
      boost::throw_exception(promise_already_satisfied());
    }

    task->apply(boost::move(args)...);
  }

  template <typename F>
  void set_wait_callback(F f) {
    task->set_wait_callback(f, this);
  }
}; // packaged_task
} // namespace boost

namespace boost {
namespace container {

  template <typename R, typename Allocator>
  struct uses_allocator<boost::packaged_task<R>, Allocator> : true_type {};

} // namespace container
} // namespace boost

namespace boost {

BOOST_THREAD_DCL_MOVABLE_BEG(T)
packaged_task<T>
BOOST_THREAD_DCL_MOVABLE_END

namespace detail {

template <typename S, typename F>
BOOST_THREAD_FUTURE<S> make_future_async_shared_state(
  BOOST_THREAD_FWD_REF(F) f) {
  boost::shared_ptr<future_async_shared_state<S, F> > h(
    new future_async_shared_state<S, F>());
  h->init(boost::forward<F>(f));
  return BOOST_THREAD_FUTURE<S>(h);
}

template <typename S, typename F>
BOOST_THREAD_FUTURE<S> make_future_deferred_shared_state(
  BOOST_THREAD_FWD_REF(F) f) {
  boost::shared_ptr<future_deferred_shared_state<S, F> > h(
    new future_deferred_shared_state<S, F>(boost::forward<F>(f)));
  return BOOST_THREAD_FUTURE<S>(h);
}
} // namespace detail

template <typename S, typename... Args>
BOOST_THREAD_FUTURE<S> async(
  launch policy,
  S(*f)(BOOST_THREAD_FWD_REF(Args)...),
  BOOST_THREAD_FWD_REF(Args) ...args) {
  typedef S(*F)(BOOST_THREAD_FWD_REF(Args)...);
  typedef boost::detail::invoker<
    typename boost::decay<F>::type,
    typename boost::decay<Args>::type...> BF;
  typedef typename BF::result_type result_type;

  if (underlying_cast<int>(policy) & int(boost::launch::async)) {
    return BOOST_THREAD_MAKE_RV_REF(
      boost::detail::make_future_async_shared_state<result_type>(
        BF(f, boost::thread_detail::decay_copy(
          boost::forward<Args>(args))...)));
  } else if (underlying_cast<int>(policy) & int(boost::launch::deferred)) {
    return BOOST_THREAD_MAKE_RV_REF(
      boost::detail::make_future_deferred_shared_state<result_type>(
        BF(f, boost::thread_detail::decay_copy(
          boost::forward<Args>(args))...)));
  } else {
    std::terminate();
  }
} // async

template <typename F, typename... Args>
BOOST_THREAD_FUTURE<typename boost::result_of<
  typename decay<F>::type(
    typename decay<Args>::type...)>::type> async(
  launch policy, BOOST_THREAD_FWD_REF(F) f,
  BOOST_THREAD_FWD_REF(Args) ...args) {
  typedef boost::detail::invoker<
    typename decay<F>::type,
    typename decay<Args>::type...> BF;
  typedef typename BF::result_type S;

  if (underlying_cast<int>(policy) & int(boost::launch::async)) {
    return BOOST_THREAD_MAKE_RV_REF(
      boost::detail::make_future_async_shared_state<S>(
        BF(
          boost::thread_detail::decay_copy(boost::forward<F>(f)),
          boost::thread_detail::decay_copy(boost::forward<Args>(args))...
        )
      )
    );
  } else {
    std::terminate();
  }
} // async

namespace detail {

template <typename S, typename F>
struct shared_state_nullary_task {
  typedef boost::shared_ptr<boost::detail::shared_state_base> storage_type;
  storage_type st_;
  F f_;

  shared_state_nullary_task(storage_type st, BOOST_THREAD_FWD_REF(F) f) :
    st_(st), f_(boost::move(f)) {}

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

  shared_state_nullary_task(BOOST_THREAD_RV_REF(shared_state_nullary_task) s) :
    st_(s.st_), f_(boost::move(s.f_)) {
    s.st_.reset();
  }

  shared_state_nullary_task& operator=(
    BOOST_THREAD_RV_REF(shared_state_nullary_task) s) {
    if (this != &s) {
      st_ = s.st_;
      f_ = boost::move(s.f_);
      s.st_.reset();
    }
    return *this;
  }

  void operator()() {
    boost::shared_ptr<shared_state<S> > st_ =
      static_pointer_cast<shared_state<S> >(st_);

    try {
      st_->mark_finished_with_result(f_());
    } catch (...) {
      st_->mark_exceptional_finish();
    }
  }

  ~shared_state_nullary_task() {}
}; // shared_state_nullary_task

template <typename F>
struct shared_state_nullary_task<void, F> {
  typedef boost::shared_ptr<boost::detail::shared_state_base> storage_type;
  storage_type st_;
  F f_;

  shared_state_nullary_task(storage_type st, BOOST_THREAD_FWD_REF(F) f) :
    st_(st), f_(boost::move(f)) {}

  BOOST_THREAD_COPYABLE_AND_MOVABLE(shared_state_nullary_task)
  shared_state_nullary_task(shared_state_nullary_task const& s) :
    st_(s.st_), f_(s.f_) {}

  shared_state_nullary_task& operator=(
    BOOST_THREAD_COPY_ASSIGN_REF(shared_state_nullary_task) s) {
    if (this != &s) {
      st_ = s.st_;
      f_ = s.f_;
    }
    return *this;
  }

  shared_state_nullary_task(
    BOOST_THREAD_RV_REF(shared_state_nullary_task) s) BOOST_NOEXCEPT :
    st_(s.st_), f_(boost::move(s.f_)) {
    s.st_.reset();
  }

  shared_state_nullary_task& operator=(
    BOOST_THREAD_RV_REF(shared_state_nullary_task) s) BOOST_NOEXCEPT {
    if (this != &s) {
      st_ = s.st_;
      f_ = boost::move(s.f_);
      s.st_.reset();
    }
    return *this;
  }

  void operator()() {
    boost::shared_ptr<shared_state<void> > st_ =
      static_pointer_cast<shared_state<void> >(st_);

    try {
      f_();
      st_->mark_finished_with_result();
    } catch (...) {
      st_->mark_exceptional_finish();
    }
  }
}; // shared_state_nullary_task

} // namespace detail

BOOST_THREAD_DCL_MOVABLE_BEG2(R, F)
boost::detail::shared_state_nullary_task<R, F>
BOOST_THREAD_DCL_MOVABLE_END

namespace detail {

template <typename S>
struct future_executor_shared_state : shared_state<S> {
  typedef shared_state<S> base_type;

  future_executor_shared_state() {}

  template <typename E, typename F>
  void init(E& ex, BOOST_THREAD_FWD_REF(F) f) {
    typedef typename decay<F>::type result_type;
    this->set_executor_policy(
      boost::executor_ptr_type(new executor_ref<E>(ex)));
    shared_state_nullary_task<S, result_type> t(
      this->shared_from_this(), boost::forward<F>(f));
    ex.submit(boost::move(f));
  }

  ~future_executor_shared_state() {}
}; // future_executor_shared_size

template <typename E, typename F, typename S>
BOOST_THREAD_FUTURE<S> make_future_executor_shared_state(
  E& ex, BOOST_THREAD_FWD_REF(F) f) {
  boost::shared_ptr<future_executor_shared_state<S> > h(
    new future_executor_shared_state<S>());
  h->init(ex, boost::forward<F>(f));
  return BOOST_THREAD_FUTURE<S>(h);
} // make_future_executor_shared_state
} // namespace detail

template <typename E, typename S, typename... Args>
BOOST_THREAD_FUTURE<S> async(
  E& ex,
  S(*f)(BOOST_THREAD_FWD_REF(Args)...),
  BOOST_THREAD_FWD_REF(Args) ...args) {
  typedef S(*F)(BOOST_THREAD_FWD_REF(Args)...);
  typedef boost::detail::invoker<
    typename decay<F>::type,
    typename decay<Args>::type...> BF;
  typedef typename BF::result_type result_type;

  return BOOST_THREAD_MAKE_RV_REF(
    boost::detail::make_future_executor_shared_state<result_type>(ex, BF(
      f, boost::thread_detail::decay_copy(boost::forward<Args>(args))...
    ))
  );
} // async

template <typename S, typename... Args>
BOOST_THREAD_FUTURE<S> async(
  S(*f)(BOOST_THREAD_MAKE_RV_REF(Args)...),
  BOOST_THREAD_FWD_REF(Args) ...args) {
  return BOOST_THREAD_MAKE_RV_REF(async(boost::launch(boost::launch::any), f,
    boost::forward<Args>(args)...));
} // async

template <typename F, typename... Args>
BOOST_THREAD_FUTURE<typename boost::result_of<
  typename decay<F>::type(
    typename decay<Args>::type...)>::type> async(
  BOOST_THREAD_FWD_REF(F) f,
  BOOST_THREAD_FWD_REF(Args)... args) {
  return BOOST_THREAD_MAKE_RV_REF(
    async(boost::launch(boost::launch::any),
    boost::forward<F>(f),
    boost::forward<Args>(args)...));
} // async

template <typename T>
BOOST_THREAD_FUTURE<
  typename boost::decay<T>::type> make_future(BOOST_THREAD_FWD_REF(T) value) {
  typedef typename boost::decay<T>::type future_value_type;
  promise<future_value_type> p;
  p.set_value(boost::forward<future_value_type>(value));
  return BOOST_THREAD_MAKE_RV_REF(p.get_future());
} // make_future

inline BOOST_THREAD_FUTURE<void> make_future() {
  promise<void> p;
  p.set_value();
  return BOOST_THREAD_MAKE_RV_REF(p.get_future());
}  // make_future

namespace detail {

template <typename T>
struct deduced_type_impl {
  typedef T type;
};

template <typename T>
struct deduced_type_impl<boost::reference_wrapper<T> const> {
  typedef T& type;
};

template <typename T>
struct deduced_type_impl<boost::reference_wrapper<T> > {
  typedef T& type;
};

template <typename T>
struct deduced_type {
  typedef typename deduced_type_impl<typename decay<T>::type>::type type;
};
} // namespace detail

template <int = 0, int..., typename T>
BOOST_THREAD_FUTURE<
  typename boost::detail::deduced_type<T>::type> make_ready_future(
    BOOST_THREAD_FWD_REF(T) value) {
  typedef typename boost::detail::deduced_type<T>::type future_value_type;
  promise<future_value_type> p;
  p.set_value(boost::forward<T>(value));
  return BOOST_THREAD_MAKE_RV_REF(p.get_future());
} // make_ready_future

template <typename T>
BOOST_THREAD_FUTURE<T> make_ready_future(
  typename boost::remove_reference<T>::type& value) {
  promise<T> p;
  p.set_value(value);
  return p.get_future();
} // make_ready_future

template <typename T>
BOOST_THREAD_FUTURE<T> make_ready_future(BOOST_THREAD_FWD_REF(
  typename boost::remove_reference<T>::type) value) {
  promise<T> p;
  p.set_value(boost::forward<
    typename boost::remove_reference<T>::type>(value));
  return p.get_future();
} // make_ready-future

template <typename T, typename... Args>
BOOST_THREAD_FUTURE<T> make_ready_future(Args&& ...args) {
  promise<T> p;
  p.emplace(boost::forward<Args>(args)...);
  return p.get_future();
} // make_ready_future

inline BOOST_THREAD_FUTURE<void> make_ready_future() {
  promise<void> p;
  p.set_value();
  return p.get_future();
} // make_ready_future

template <typename T, typename U>
BOOST_THREAD_FUTURE<T> make_ready_no_decay_future(U value) {
  typedef T future_value_type;
  promise<future_value_type> p;
  p.set_value(value);
  return BOOST_THREAD_MAKE_RV_REF(p.get_future());
} // make_ready_no_decay_future

template <typename T>
BOOST_THREAD_FUTURE<T> make_exceptional_future(boost::exception_ptr e) {
  promise<T> p;
  p.set_exception(e);
  return BOOST_THREAD_MAKE_RV_REF(p.get_future());
} // make_exceptional_future

#if 0
  template <typename C>
  make_future<C c) -> BOOST_THREAD_FUTURE<decltype(c())> {
    typedef decltype(c()) T;
    promise<T> p;
    try {
      p.set_value(c());
    } catch (...) {
      p.set_exception(boost::current_exception());
    }
    return BOOST_THREAD_MAKE_RV_REF(p.get_future());
  }
#endif

template <typename T>
shared_future<typename decay<T>::type> make_shared_future(
  BOOST_THREAD_FWD_REF(T) value) {
  typedef typename decay<T>::type future_type;
  promise<future_type> p;
  p.set_value(boost::forward<T>(value));
  return BOOST_THREAD_MAKE_RV_REF(p.get_future().share());
} // make_shared_future

inline shared_future<void> make_shared_future() {
  promise<void> p;
  return BOOST_THREAD_MAKE_RV_REF(p.get_future().share());
} // make_shared_future

namespace detail {

template <typename F, typename S, typename C, typename St = shared_state<S> >
struct continuation_shared_state : St {
  F parent;
  C continuation;

  continuation_shared_state(
    BOOST_THREAD_RV_REF(F) f, BOOST_THREAD_FWD_REF(C) c) :
    parent(boost::move(f)), continuation(boost::move(c)) {}

  void init(boost::unique_lock<boost::mutex>& lock) {
    parent.future_->set_continuation_ptr(this->shared_from_this(), lock);
  }

  void call() {
    try {
      this->mark_finished_with_result(
        this->continuation(boost::move(this->parent)));
    } catch (...) {
      this->mark_exceptional_finish();
    }
    this->parent = F();
  }

  void call(boost::unique_lock<boost::mutex>& lock) {
    try {
      relocker lock(lock);
      S r = this->continuation(boost::move(this->parent));
      this->parent = F();
      lock.lock();
      this->mark_finish_with_result_internal(boost::move(r), lock);
    } catch (...) {
      this->mark_exceptional_finish_internal(boost::current_exception(), lock);
      relocker lock(lock);
      this->parent = F();
    }
  }

  static void run(boost::shared_ptr<boost::detail::shared_state_base> that) {
    continuation_shared_state* that_ =
      static_cast<continuation_shared_state*>(that.get());
    that_->call();
  }

  ~continuation_shared_state() {}
}; // continuation_shared_state

template <typename F, typename C, typename St>
struct continuation_shared_state<F, void, C, St> : St {
  F parent;
  C continuation;

  continuation_shared_state(
    BOOST_THREAD_RV_REF(F) f, BOOST_THREAD_FWD_REF(C) c) :
    parent(boost::move(f)), continuation(boost::move(c)) {}

  void init(boost::unique_lock<boost::mutex>& lock) {
    parent.future_->set_continuation_ptr(this->shared_from_this(), lock);
  }

  void call() {
    try {
      this->continuation(boost::move(this->parent));
      this->mark_finished_with_result();
    } catch (...) {
      this->mark_exceptional_finish();
    }
    this->parent = F();
  }

  void call(boost::unique_lock<boost::mutex>& lock) {
    try {
      relocker lock(lock);
      this->continuation(boost::move(this->parent));
      this->parent = F();
      this->mark_finished_with_result_internal(lock);
    } catch (...) {
      this->mark_exceptional_finish_internal(boost::current_exception(), lock);
      relocker lock(lock);
      this->parent = F();
    }
  }

  static void run(boost::shared_ptr<boost::detail::shared_state_base> that) {
    continuation_shared_state* that_ =
      static_cast<continuation_shared_state*>(that.get());
    that_->call();
  }

  ~continuation_shared_state() {}
}; // continuation_shared_state

template <typename F, typename S, typename C>
struct future_async_continuation_shared_state :
  boost::detail::continuation_shared_state<F, S, C,
    boost::detail::future_async_shared_state_base<S> > {
  typedef boost::detail::continuation_shared_state<F, S, C,
    boost::detail::future_async_shared_state_base<S> > base_type;

  future_async_continuation_shared_state(
    BOOST_THREAD_RV_REF(F) f, BOOST_THREAD_FWD_REF(C) c) :
    base_type(boost::move(f), boost::forward<C>(c)) {}

  void launch_continuation() {
    boost::lock_guard<boost::mutex> lock(this->mutex);
    this->th_ = boost::thread(&future_async_continuation_shared_state::run,
      boost::static_shared_from_this(this));
  }
}; // future_async_continuation_shared_state

template <typename F, typename S, typename C>
struct future_sync_continuation_shared_state :
  boost::detail::continuation_shared_state<
    F, S, C, boost::detail::shared_state<S> > {
  typedef boost::detail::continuation_shared_state<
    F, S, C, boost::detail::shared_state<S> > base_type;

  future_sync_continuation_shared_state(
    BOOST_THREAD_RV_REF(F) f, BOOST_THREAD_FWD_REF(C) c) :
    base_type(boost::move(f), boost::forward<C>(c)) {}

  void launch_continuation() {
    this->call();
  }
}; // future_sync_continuation_shared_state

template <typename Ex>
struct run_it {
  boost::shared_ptr<Ex> that_;

  BOOST_THREAD_COPYABLE_AND_MOVABLE(run_it)
  run_it(run_it const& ex) : that_(ex.that_) {}

  run_it& operator=(BOOST_THREAD_COPY_ASSIGN_REF(run_it) ex) {
    if (this != &ex) {
      that_ = ex.that_;
    }
    return *this;
  }

  run_it(BOOST_THREAD_RV_REF(run_it) ex) BOOST_NOEXCEPT : that_(ex.that_) {
    ex.that_.reset();
  }

  run_it& operator=(BOOST_THREAD_RV_REF(run_it) ex) BOOST_NOEXCEPT {
    if (this != &ex) {
      that_ = ex.that_;
      ex.that_.reset();
    }
    return *this;
  }

  run_it(boost::shared_ptr<Ex> that) : that_(that) {}

  void operator()() {
    that_->run(that_);
  }
}; // run_it
} // namespace detail

BOOST_THREAD_DCL_MOVABLE_BEG(F)
detail::run_it<F>
BOOST_THREAD_DCL_MOVABLE_END

namespace detail {

/* future_executor_continuation_shared_state */
template <typename F, typename S, typename C>
struct future_executor_continuation_shared_state :
  boost::detail::continuation_shared_state<F, S, C> {
  typedef boost::detail::continuation_shared_state<F, S, C> base_type;

  future_executor_continuation_shared_state(
    BOOST_THREAD_RV_REF(F) f, BOOST_THREAD_FWD_REF(C) c) :
    base_type(boost::move(f), boost::forward<C>(c)) {}

  template <typename Ex>
  void init(boost::unique_lock<boost::mutex>& lock, Ex& ex) {
    this->set_executor_policy(
      boost::executor_ptr_type(new executor_ref<Ex>(ex)), lock);
    this->base_type::init(lock);
  }

  void launch_continuation() {
    run_it<base_type> f(boost::static_shared_from_this(this));
    this->get_executor()>submit(boost::move(f));
  }

  ~future_executor_continuation_shared_state() {}
}; // future_executor_continuation_shared_state

/* shared_future_async_continuation_shared_state */
template <typename F, typename S, typename C>
struct shared_future_async_continuation_shared_state :
  boost::detail::continuation_shared_state<
    F, S, C, boost::detail::future_async_shared_state_base<S> > {
  typedef boost::detail::continuation_shared_state<
    F, S, C, boost::detail::future_async_shared_state_base<S> > base_type;

  shared_future_async_continuation_shared_state(
    F f, BOOST_THREAD_FWD_REF(C) c) :
    base_type(boost::move(f), boost::forward<C>(c)) {}

  void launch_continuation() {
    boost::lock_guard<boost::mutex> lock(this->mutex);
    this->th_ = boost::thread(
      &base_type::run, boost::static_shared_from_this(this));
  }
}; // shared_future_async_continuation_shared_state

/*  shared_future_sync_continuation_shared_state */
template <typename F, typename S, typename C>
struct shared_future_sync_continuation_shared_state :
  boost::detail::continuation_shared_state<
    F, S, C, boost::detail::shared_state<S> > {
  typedef boost::detail::continuation_shared_state<
    F, S, C, boost::detail::shared_state<S> > base_type;

  shared_future_sync_continuation_shared_state(
    F f, BOOST_THREAD_FWD_REF(C) c) :
    base_type(boost::move(f), boost::forward<C>(c)) {}

  void launch_continuation() {
    this->call();
  }
}; // shared_future_sync_continuation_shared_state

/* shared_future_executor_continuation_shared_state */
template <typename F, typename S, typename C>
struct shared_future_executor_continuation_shared_state :
  boost::detail::continuation_shared_state<F, S, C> {
  typedef boost::detail::continuation_shared_state<F, S, C> base_type;

  shared_future_executor_continuation_shared_state(
    F f, BOOST_THREAD_FWD_REF(C) c) :
    base_type(boost::move(f), boost::forward<C>(c)) {}

  template <typename Ex>
  void init(boost::unique_lock<boost::mutex>& lock, Ex& ex) {
    this->set_executor_policy(
      boost::executor_ptr_type(new executor_ref<Ex>(ex)), lock);
    this->base_type::init();
  }

  void launch_continuation() {
    run_it<base_type> f(boost::static_shared_from_this(this));
    this->get_executor()->submit(boost::move(f));
  }

  ~shared_future_executor_continuation_shared_state() {}
}; // shared_future_executor_continuation_shared_state

/* future_deferred_continuation */
template <typename F, typename S, typename C>
struct future_deferred_continuation_shared_state :
  boost::detail::continuation_shared_state<F, S, C> {
  typedef boost::detail::continuation_shared_state<F, S, C> base_type;

  future_deferred_continuation_shared_state(
    BOOST_THREAD_RV_REF(F) f, BOOST_THREAD_FWD_REF(C) c) :
    base_type(boost::move(f), boost::forward<C>(c)) {
    this->set_deferred();
  }

  virtual void execute(boost::unique_lock<boost::mutex>& lock) {
    this->parent.wait();
    this->call(lock);
  }

  virtual void launch_continuation() {}
}; // future_deferred_continuation()

/* shared_future_deferred_continuation_shared_state */
template <typename F, typename S, typename C>
struct shared_future_deferred_continuation_shared_state :
  boost::detail::continuation_shared_state<F, S, C> {
  typedef boost::detail::continuation_shared_state<F, S, C> base_type;

  shared_future_deferred_continuation_shared_state(
    F f, BOOST_THREAD_FWD_REF(C) c) :
    base_type(boost::move(f), boost::forward<C>(c)) {
    this->set_deferred();
  }

  virtual void execute(boost::unique_lock<boost::mutex>& lock) {
    this->parent.wait();
    this->call(lock);
  }

  virtual void launch_continuation() {}
}; // shared_future_deferred_continuation_shared_state

/*  make_future_async_continuation_shared_state */
template <typename F, typename S, typename C>
BOOST_THREAD_FUTURE<S> make_future_async_continuation_shared_state(
  boost::unique_lock<boost::mutex>& lock,
  BOOST_THREAD_RV_REF(F) f,
  BOOST_THREAD_FWD_REF(C) c) {
  typedef typename decay<C>::type callback_type;

  boost::shared_ptr<
    boost::detail::future_async_continuation_shared_state<
      F, S, callback_type> > h(
        new boost::detail::future_async_continuation_shared_state<
          F, S, callback_type>(
            boost::move(f), boost::forward<C>(c)));
  h->init(lock);

  return BOOST_THREAD_FUTURE<S>(h);
} // make_future_async_continuation_shared_state

/*  make_future_sync_continuation_shared_state */
template <typename F, typename S, typename C>
BOOST_THREAD_FUTURE<S> make_future_sync_continuation_shared_state(
  boost::unique_lock<boost::mutex>& lock,
  BOOST_THREAD_RV_REF(F) f,
  BOOST_THREAD_FWD_REF(C) c) {
  typedef typename decay<C>::type callback_type;
  boost::shared_ptr<
    boost::detail::future_sync_continuation_shared_state<
      F, S, callback_type> > h(
        new boost::detail::continuation_shared_state<
          F, S, callback_type>(
            boost::move(f), boost::forward<C>(c)));
  h->init(lock);

  return BOOST_THREAD_FUTURE<S>(h);
} // make_future_sync_continuation_shared_state

/*  make_future_deferred_continuation_shared_state */
template <typename F, typename S, typename C>
BOOST_THREAD_FUTURE<S> make_future_deferred_continuation_shared_state(
  boost::unique_lock<boost::mutex>& lock,
  BOOST_THREAD_RV_REF(F) f,
  BOOST_THREAD_FWD_REF(C) c) {
  typedef typename decay<C>::type callback_type;

  boost::shared_ptr<
    boost::detail::future_deferred_continuation_shared_state<
      F, S, callback_type> > h(
        new boost::detail::future_deferred_continuation_shared_state<
          F, S, callback_type>(
            boost::move(f), boost::forward<C>(c)));
  h->init(lock);

  return BOOST_THREAD_FUTURE<S>(c);
} // make_future_deferred_continuation_shared_state

/*  make_future_executor_continuation_shared_state */
template <typename E, typename F, typename S, typename C>
BOOST_THREAD_FUTURE<S> make_future_executor_continuation_shared_state(
  E& ex,
  boost::unique_lock<boost::mutex>& lock,
  BOOST_THREAD_RV_REF(F) f,
  BOOST_THREAD_FWD_REF(C) c) {
  typedef typename decay<C>::type callback_type;
  boost::shared_ptr<
    boost::detail::future_executor_continuation_shared_state<
      F, S, callback_type> > h(
        new boost::detail::future_executor_continuation_shared_state<
          F, S, callback_type>(
            boost::move(f), boost::forward<C>(c)));
  h->init(lock);

  return BOOST_THREAD_FUTURE<S>(h);
} // make_future_executor_continuation_shared_state

} // namespace detail
} // namespace boost
