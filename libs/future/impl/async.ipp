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
  } else if (boost::underlying_cast<int>(policy) &&
             int(boost::launch::deferred)) {
    return BOOST_THREAD_MAKE_RV_REF(
      boost::detail::make_future_deferred_shared_state<result_type>(
        boost::thread_detail::decay_copy(
          boost::forward<F>(f)),
        boost::thread_detail::decay_copy(
          boost::forward<As>(as)));
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
#ifdef BOOST
} // boost
#endif // ASYNC_IPP
