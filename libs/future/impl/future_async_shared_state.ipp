#ifndef FUTURE_ASYNC_SHARED_STATE_IPP
#define FUTURE_ASYNC_SHARED_STATE_IPP
#include "../include/futures.hpp"

namespace boost {
namespace detail {

template <typename S, typename F>
struct future_async_shared_state :
  boost::detail::future_async_shared_state_base<S> {
  void init(
    BOOST_THREAD_FWD_REF(F) f) {

#ifdef BOOST_THREAD_FUTURE_BLOCKING
    this->thr_ = boost::thread(
      &future_async_shared_state::run,
      boost::static_shared_from_this(this),
      boost::forward<F>(f));
#else
    boost::thread(
      &future_async_shared_state::run,
      boost::static_shared_from_this(this),
      boost::forward<F>(f)).detach();
#endif // BOOST_THREAD_FUTURE_BLOCKING

  }

  static void run(
    boost::shared_ptr<future_async_shared_state> that,
    BOOST_THREAD_FWD_REF(F) f) {
    try {
      that->mark_finished_with_result(f());
    } catch (...) {
      that->mark_exceptional_finish();
    }
  }
};

template <typename F>
struct future_async_shared_state<void, F> :
  public boost::detail::future_async_shared_state_base<void> {
  void init(
    BOOST_THREAD_FWD_REF(F) f) {

#ifdef BOOST_THREAD_FUTURE_BLOCKING
  this->thr_ = boost::thread(
    &future_async_shared_state::run,
    boost::static_shared_from_this(this),
    boost::move(f));
#else
  boost::thread(
    &future_async_shared_state::run,
    boost::static_shared_from_this(this),
    boost::move(f)).detach();
#endif // BOOST_THREAD_FUTURE_BLOCKING

  }

  static void run(
    boost::shared_ptr<future_async_shared_state> that,
    BOOST_THREAD_FWD_REF(F) f) {
    try {
      f();
      that->mark_finished_with_result();
    } catch (...) {
      that->mark_exceptional_finish();
    }
  }
};

template <typename S, typename F>
struct future_async_shared_state<S&, F> :
  boost::detail::future_async_shared_state_base<S&> {
  void init(
    BOOST_THREAD_FWD_REF(F) f) {

#ifdef BOOST_THREAD_FUTURE_BLOCKING
    this->thr_ = boost::thread(
      &future_async_shared_state::run,
      boost::static_shared_from_this(this),
      boost::move(f));
#else
    boost::thread(
      &future_async_shared_state::run,
      boost::static_shared_from_this(this),
      boost::move(f));
#endif // BOOST_THREAD_FUTURE_BLOCKING

  }

  static void run(
    boost::shared_ptr<future_async_shared_state> that,
    BOOST_THREAD_FWD_REF(F) f) {
    try {
      that->mark_finished_with_result(f());
    } catch (...) {
      that->mark_exceptional_finish();
    }
  }
};

} // detail
} // boost

#endif // FUTURE_ASYNC_SHARED_STATE_IPP
