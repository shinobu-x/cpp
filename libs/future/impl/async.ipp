#ifndef ASYNC_IPP
#define ASYNC_IPP

#include "../include/futures.hpp"

namespace boost {

#ifdef BOOST_THREAD_RVALUE_REFERENCES_DONT_MATCH_FUNCTION_PTR
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
template <typename R, typename... As>
BOOST_THREAD_FUTURE<R> async(
  boost::launch policy,
  R(*f)(BOOST_THREAD_FWD_REF(As)...),
  BOOST_THREAD_FWD_REF(As) ...as) {
  typedef R(*F)(BOOST_THREAD_FWD_REF(As)...);
  typedef boost::detail::invoker<
    typename boost::decay<F>::type,
    typename boost::decay<As>::type...> callback_type;
  typedef typename callback_type::result_type result_type;

  if (boost::underlying_cast<int>(policy) &&
      int(boost::launch::async)) {
    return BOOST_THREAD_MAKE_RV_REF(
      boost::detail::make_future_async_shared_state<result_type>(
        callback_type(
          f,
          boost::thread_detail::decay_copy(
            boost::forward<As>(as))...)));
  } else if (boost::underlying_cast<int>(policy) &&
             int(boost::launch::deferred)) {
    return BOOST_THREAD_MAKE_RV_REF(
      boost::detail::make_future_deferred_shared_state<result_type>(
        callback_type(
          f,
          boost::thread_detail::decay_copy(
            boost::forward<As>(as))...)));
  } else {
    std::terminate();
  }
}
#else // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
template <typename R>
BOOST_THREAD_FUTURE<R> async(
  boost::launch policy,
  R(*f)()() {
#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
  typedef boost::packaged_task<R()> packaged_task_type;
#else // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
  typedef boost::packaged_task<R> packaged_task_type;
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK

  if (boost::underlying_cast<int>(policy) &&
      int(boost::launch::async)) {
    packaged_task_type task_type(f);
    BOOST_THREAD_FUTURE<R> r =
      BOOST_THREAD_MAKE_RV_REF(task_type.get_future());
    r.set_async();
    return boost::move(r);
  } else if (boost::underlying_cast<int>(policy) &&
             boost::launch::deferred) {
    std::terminate();
  } else {
    std::terminate();
  }
}
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
#endif // BOOST_THREAD_RVALUE_REFERENCES_DONT_MATCH_FUNCTION_PTR

#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
template <typename F, typename... As>
BOOST_THREAD_FUTURE<
  typename boost::result_of<
    typename boost::decay<F>::type(
      typename boost::decay<As>::type...)>::type> async(
  boost::launch policy,
  BOOST_THREAD_FWD_REF(F) f,
  BOOST_THREAD_FWD_REF(As) ...as) {
  typedef boost::detail::invoker<
    typename boost::decay<F>::type,
    typename boost::decay<As>::type...> callback_type;
  typedef typename callback_type::result_type result_type;

  if (boost::underlying_cast<int>(policy) &&
      int(boost::launch::async)) {
    return BOOST_THREAD_MAKE_RV_REF(
      boost::detail::make_future_async_shared_state<result_type>(
        callback_type(
          boost::thread_detail::decay_copy(
            boost::forward<F>(f)),
          boost::thread_detail::decay_copy(
            boost::forward<As>(as))...)));
  }  else if (boost::underlying_cast<int>(policy) &&
             int(boost::launch::deferred)) {
    return BOOST_THREAD_MAKE_RV_REF(
      boost::detail::make_future_deferred_shared_state<result_type>(
        callback_type(
          boost::thread_detail::decay_copy(
            boost::forward<F>(f)),
          boost::thread_detail::decay_copy(
            boost::forward<As>(as))...)));
  } else {
    std::terminate();
  }

}

#else // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
template <typename F>
BOOST_THREAD_FUTURE<
  typename boost::result_of<
    typename boost::decay<F>::type()>::type> async(
  boost::launch policy,
  BOOST_THREAD_FWD_REF(F) f) {
  typedef typename boost::result_of<
    typename boost::decay<F>::type()>::type task_type;
#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
  typedef boost::packaged_task<task_type()> packaged_task_type;
#else // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
  typedef boost::packaged_task<task_type> packaged_task_type;
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK

  if (boost::underlying_cast<int>(policy) &&
      int(boost::launch::async)) {
    packaged_task_type task(
      boost::forward<F>(f));
    BOOST_THREAD_FUTURE<R> r = task.get_future();
    r.set_async();
    boost::thread(boost::move(task)).detach();
    return boost::move(r);
  } else if (boost::underlying_cast<int>(policy) &&
             int(boost::launch::deferred)) {
    std::terminate();
  } else {
    std::terminate();
  }
}

#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
#if defined(BOOST_THREAD_PROVIDES_INVOKE) &&                                  \
   !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATE) &&                              \
   !defined(BOOST_NO_CXX11_HDR_TUPLE)
#ifdef BOOST_THREAD_RVALUE_REFERENCES_DONT_MATCH_FUNCTION_PTR
template <typename Ex, typename R, typename... As>
BOOST_THREAD_FUTURE<R> async(
  Ex& ex,
  R(*f)(BOOST_THREAD_FWD_REF(As)...),
  BOOST_THREAD_FWD_REF(As) ...as) {
  typedef R(*F)(BOOST_THREAD_FWD_REF(As)...);
  typedef boost::detail::invoker<
    typename boost::decay<F>::type,
    typename boost::decay<As>::type...> callback_type;
  typedef typename callback_type::result_type result_type;

  return BOOST_THREAD_MAKE_RV_REF(
    boost::detail::make_future_executor_shared_state<result_type>(
      ex,
      callback_type(
        boost::thread_detail::decay_copy(
          boost::forward<F>(f)),
        boost::thread_detail::decay_copy(
          boost::forward<As>(as))...
    )));
}
#endif // BOOST_THREAD_RVALUE_REFERENCES_DONT_MATCH_FUNCTION_PTR
template <typename Ex, typename F, typename... As>
BOOST_THREAD_FUTURE<
  typename boost::result_of<
    typename boost::decay<F>::type(
      typename boost::decay<As>::type...)>::type>
  async(
    Ex& ex,
    BOOST_THREAD_FWD_REF(F) f,
    BOOST_THREAD_FWD_REF(As) ...as) {
  typedef boost::detail::invoker<
    typename boost::decay<F>::type,
    typename boost::decay<As>::type...> callback_type;
  typedef typename callback_type::result_type result_type;

  return BOOST_THREAD_MAKE_RV_REF(
    boost::detail::make_future_executor_shared_state<result_type>(
      ex,
      callback_type(
        boost::thread_detail::decay_copy(
          boost::forward<F>(f)),
        boost::thread_detail::decay_copy(
          boost::forward<As>(as))...
    )));
}
#else // BOOST_THREAD_PROVIDES_INVOKE
      // BOOST_NO_CXX11_VARIADIC_TEMPLATE
      // BOOST_NO_CXX11_HDR_TUPLE
#ifdef BOOST_THREAD_RVALUE_REFERENCES_DONT_MATCH_FUNCTION_PTR
template <typename Ex, typename R>
BOOST_THREAD_FUTURE<R> async(
  Ex& ex,
  R(*f)()) {
  typedef R(*F)();
  typedef boost::detail::invoker<F> callback_type;
  typedef typename callback_type::result_type result_type;

  return BOOST_THREAD_MAKE_RV_REF(
    boost::detail::make_future_executor_shared_state<result_type>(
      ex,
      callback_type(f)));
}

template <typename Ex, typename R, typename A>
BOOST_THREAD_FUTURE<R> async(
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
        boost::thread_detail::decay_copy(boost::forward<A>(a));
    )));
}
#endif // BOOST_THREAD_RVALUE_REFERENCES_DONT_MATCH_FUNCTION_PTR
template <typename Ex, typename F>
BOOST_THREAD_FUTURE<
  typename boost::result_of<
    typename boost::decay<F>::type()>::type async(
  Ex& ex,
  BOOST_THREAD_FWD_REF(F) f) {
  typedef boost::detail::invoker<
    typename boost::decay<F>::type> callback_type;
  typedef typename callback_type::result_type result_type;

  return BOOST_THREAD_MAKE_RV_REF(
    boost::detail::make_future_executor_shared_state<result_type>(
      ex,
      callback_type(
        boost::thread_detail::decay_copy(
          boost::forward<F>(f)))));
}

template <typename Ex, typename F, typename A1>
BOOST_THREAD_FUTURE<
  typename boost::result_of<
    typename boost::decay<F>::type(
      typename boost::decay<A1>::type)>::type async(
  Ex& ex,
  BOOST_THREAD_FWD_REF(F) f,
  BOOST_THREAD_FWD_REF(A1) a1) {
  typedef boost::detail::invoker<
    typename boost::decay<F>::type,
    typename boost::decay<A1>::type> callback_type;
  typedef typename callback_type::result_type result_type;

  return BOOST_THREAD_MAKE_RV_REF(
    boost::detail::make_future_executor_shared_state(
      ex,
      callback_type(
        boost::thread_detail::decay_copy(
          boost::forward<F>(f)),
        boost::thread_detail::decay_copy(
          boost::forward<A1>(a1)))));
}

template <typename Ex, typename F, typename T1, typename T2>
BOOST_THREAD_FUTURE<
  typename boost::result_of<
    typename boost::decay<F>::type(
      typename boost::decay<T1>::type,
      typename boost::decay<T2>::type)>::type> async(
  Ex& ex,
  BOOST_THREAD_FWD_REF(F) f,
  BOOST_THREAD_FWD_REF(A1) a1,
  BOOST_THREAD_FWD_REF(A2) a2) {
  typedef boost::detail::invoker<
    typename boost::decay<F>::type,
    typename boost::decay<A1>::type,
    typename boost::decay<A2>::type> callback_type;
  typedef typename callback_type::result_type result_type;

  return BOOST_THREAD_MAKE_RV_REF(
    boost::detail::make_future_executor_shared_state<result_type>(
      ex,
      boost::thread_detail::decay_copy(
        boost::forward<F>(f)),
      boost::thread_detail::decay_copy(
        boost::forward<A1>(a1)),
      boost::thread_detail::decay_copy(
        boost::forward<A2>(a2))));
}
#endif // BOOST_THREAD_PROVIDES_INVOKE
       // BOOST_NO_CXX11_VARIADIC_TEMPLATE
       // BOOST_NO_CXX11_HDR_TUPLE
#endif // BOOST_THREAD_PROVIDES_EXECUTORS

#ifdef BOOST_THREAD_RVALUE_REFERENCES_DONT_MATCH_FUNCTION_PTR
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
template <typename R, typename... As>
BOOST_THREAD_FUTURE<R> async(
  R(*f)(BOOST_THREAD_FWD_REF(As)...),
  BOOST_THREAD_FWD_REF(As) ...as) {
  return BOOST_THREAD_MAKE_RV_REF(
    boost::async(
      boost::launch(
        boost::launch::any),
    f,
    boost::forward<As>(as)...));
}
#else // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
template <typename R>
BOOST_THREAD_FUTURE<R> async(R(*f)()) {
  return BOOST_THREAD_MAKE_RV_REF(
    boost::sync(
      boost::launch(
        boost::launch::any),
    f));
}
#endif // BOOST_THREAD_RVALUE_REFERENCES_DONT_MATCH_FUNCTION_PTR
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD

#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
template <typename F, typename... As>
BOOST_THREAD_FUTURE<
  typename boost::result_of<
    typename boost::decay<F>::type(
      typename boost::decay<Ts>::type...)>::type> async(
  BOOST_THREAD_FWD_REF(F) f,
  BOOST_THREAD_FWD_REF(
} // boost
#endif // ASYNC_IPP
