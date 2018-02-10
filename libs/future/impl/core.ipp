#ifndef CORE_IPP
#define CORE_IPP

#include "../include/futures.hpp"

namespace boost {

template <typename T>
boost::shared_ptr<T> static_shared_from_this(T* that) {
  return boost::static_pointer_cast<T>(that->shared_from_this());
}

template <typename T>
boost::shared_ptr<T const> static_shared_from_this(T const* that) {
  return boost::static_pointer_cast<T const>(that->shared_from_this());
}

class executor;

typedef boost::shared_ptr<executor> executor_ptr_type;

template <typename R>
class BOOST_THREAD_FUTURE;

template <typename R>
class shared_future;

template <typename R>
class promise;
template <typename R>
class packaged_task;

template <typename T>
struct is_future_type<BOOST_THREAD_FUTURE<T> > : boost::true_type {};

template <typename T>
struct is_future_type<shared_future<T> > : boost::true_type {};

namespace detail {

struct relocker {

  boost::unique_lock<boost::mutex>& lock_;

  relocker(boost::unique_lock<boost::mutex>& lock) : lock_(lock) {
    lock_.unlock();
  }

  ~relocker() {

    if (!lock_.owns_lock()) {
      lock_.lock();
    }

  }

  void lock() {

    if (!lock_.owns_lock()) {
      lock_.lock();
    }
  }

private:
  relocker& operator=(relocker const&);
};

class base_future {
public:
};

#if !defined _MSC_VER || _MSC_VER >= 1400
template <typename R, typename F>
BOOST_THREAD_FUTURE<R> make_future_async_shared_state(
  BOOST_THREAD_FWD_REF(F) f);

template <typename R, typename F>
BOOST_THREAD_FUTURE<R> make_future_deferred_shared_state(
  BOOST_THREAD_FWD_REF(F) f);
#endif // _MSC_VER

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
template <typename R, typename F, typename Ex>
BOOST_THREAD_FUTURE<R> make_future_executor_shared_state(
  Ex& ex,
  BOOST_THREAD_RV_REF(F) f);

template <typename Ex, typename F, typename R, typename C>
BOOST_THREAD_FUTURE<R> make_future_executor_continuation_shared_state(
  Ex& ex,
  BOOST_THREAD_RV_REF(F) f,
  BOOST_THREAD_FWD_REF(C) c);

template <typename Ex, typename F, typename R, typename C>
BOOST_THREAD_FUTURE<R> make_shared_future_executor_continuation_shared_state(
  Ex& ex,
  BOOST_THREAD_RV_REF(F) f,
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
#endif // BOOST_THREAD_PROVIDES_FUTURE_UNWRAP
} // detail
} // boost

#endif // CORE_HPP
