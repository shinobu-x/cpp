#include <boost/thread/detail/config.hpp>

#define BOOST_THREAD_FUTURE_USES_OPTIONAL
#define BOOST_THREAD_PROVIDES_EXECUTORS
#define BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
#define BOOST_THREAD_PROVIDES_FUTURE_CTOR_ALLOCATORS
#define BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY
#define BOOST_THREAD_USES_CHRONO
#define BOOST_THREAD_USES_MOVE

#include <boost/thread/condition_variable.hpp>
#include <boost/thread/detail/move.hpp>
#include <boost/thread/detail/invoker.hpp>
#include <boost/thread/detail/invoke.hpp>
#include <boost/thread/detail/is_convertible.hpp>
#include <boost/thread/exceptional_ptr.hpp>
#include <boost/thread/futures/future_error.hpp>
#include <boost/thread/futures/future_error_code.hpp>
#include <boost/thread/futures/future_status.hpp>
#include <boost/thread/futures/is_future_type.hpp>
#include <boost/thread/futures/launch.hpp>
#include <boost/thread/futures/wait_for_all.hpp>
#include <boost/thread/futures/wait_for_any.hpp>
#include <boost/thread/lock_algorithms.hpp>
#include <boost/thread/lock_types.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread_only.hpp>
#include <boost/thread/thread_time.hpp>
#include <boost/thread/executor.hpp>
#include <boost/thread/executors/generic_executor_ref.hpp>
#include <boost/optional.hpp>
#include <boost/assert.hpp>
#include <boost/bind.hpp>
#include <boost/chrono/system_clocks.hpp>
#include <boost/core/enable_if.hpp>
#include <boost/core/ref.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/exception_ptr.hpp>
#include <boost/function.hpp>
#include <boost/next_prior.hpp>
#include <boost/scoped_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <boost/throw_exception.hpp>
#include <boost/type_traits/conditional.hpp>
#include <boost/type_traits/decay.hpp>
#include <boost/type_traits/is_copy_constructible.hpp>
#include <boost/type_traits/is_fundamental.hpp>
#include <boost/type_traits/is_void.hpp>
#include <boost/utility/result_of.hpp>
#include <boost/thread/detail/memory.hpp>
#include <boost/container/scoped_allocator.hpp>
#include <boost/thread/csbl/tuple.hpp>
#include <boost/thread/csbl/vector.hpp>
#include <algorithm>
#include <list>
#include <vector>
#include <utility>

#if defined BOOST_THREAD_PROVIDES_FUTURE
#define BOOST_THREAD_FUTURE future
#else
#define BOOST_THREAD_FUTURE unique_future
#endif

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
  executor_ptr_type executor;

  virtual void launch_continuation() {}

  shared_state_base() : done_(false), is_valid_(true), is_deferred_(false),
    is_constructed_(false), policy_(boost::launch::none), continuations(),
    executor() {}

  shared_state_base(boost::exceptional_ptr const& ex) : exception(ex.ptr_),
    done_(true), is_valid_(true), is_deferred_(false),
    is_constructed_(false), policy_(boost::launch::none), continuations(),
    executor() {}

  virtual ~shared_state_base() {}

  executor_ptr_type get_executor() {
    return executor;
  }

  void set_executor_policy(executor_ptr_type ex) {
    set_executor();
    executor = ex;
  }

  void set_executor_policy(executor_ptr_type ex,
    boost::lock_guard<boost::mutex>&) {
    set_executor();
    executor = ex;
  }

  void set_executor_policy(executor_ptr_type ex,
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

  void unnotify_when_read(notify_when_ready_handle waiter) {
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

  void set_exceptional_at_thread_exit(boost::exception_ptr e) {
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

}; // shared_state_base

template <typename T>
struct shared_state : shared_state_base {
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
    shared_state_base(ex), result() {}
  ~shared_state() {}

  void mark_finish_with_result_internal(
    source_reference_type new_result,
    boost::unique_lock<boost::mutex>& lock) {
    result = new_result;
    this->mark_finish_internal(lock);
  }

  void mark_finish_with_result_internal(
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
    this->mark_finish_with_result_internal(new_result, lock);
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
    return get_storage(lock);
  }

};

} // namespace detail
} // namespace boost
