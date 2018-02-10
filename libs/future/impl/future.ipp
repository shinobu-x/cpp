#ifndef FUTURE_IPP
#define FUTURE_IPP

#include "../include/futures.hpp"

namespace boost {

template <typename T>
class BOOST_THREAD_FUTURE : public boost::detail::basic_future<T> {
  typedef boost::detail::basic_future<T> base_type;
  typedef typename base_type::future_ptr future_ptr;

  friend class boost::shared_future<T>;
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

#ifdef BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_AYN
  template <typename InputIter>
  friend typename boost::disable_if<
    boost::is_future_type<InputIter>,
    BOOST_THREAD_FUTURE<
      boost::csbl::vector<typename InputIter::value_type> > >::type
        when_all(
          InputIter begin,
          InputIter end);

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
  template <typename F, typename... Fs>
  friend BOOST_THREAD_FUTURE<
    boost::csbl::tuple<
      typename boost::decay<F>:type,
      typename boost::decay<Fs>::type...> >
        when_all(
          BOOST_THREAD_FWD_REF(F) f,
          BOOST_THREAD_FWD_REF(Fs) ...fs);
#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATES

  template <typename InputIter>
  friend typename boost::disable_if<
    boost::is_future_type<InputIter>,
    BOOST_THREAD_FUTURE<
      boost::csbl::vector<typename InputIter::value_type> > >::type
        when_any(
          InputIter begin,
          InputIter end);

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
  template <typename F, typename... Fs>
  friend BOOST_THREAD_FUTURE<
    boost::csbl::tuple<
      typename boost::decay<F>::type,
      typename boost::decay<Fs>::type...> >
        when_any(
          BOOST_THREAD_FWD_REF(F) f,
          BOOST_THREAD_FWD_REF(Fs) ...fs);
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

  typedef typename base_type::move_dest_type move_dest_type;

  // Constructor
  BOOST_THREAD_FUTURE(future_ptr future) : base_type(future() {}
} // boost

#endif // FUTURE_IPP
