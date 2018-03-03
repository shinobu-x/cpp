#ifndef FUTURE_IPP_
#define FUTURE_IPP_
#ifndef BOOST_NO_EXCEPTIONS
#include <include/futures.hpp>
#include <hpp/core.hpp>
#include <hpp/shared_state_base.hpp>
#include <hpp/shared_state.hpp>
#include <hpp/future_async_shared_state_base.hpp>
#include <hpp/future_async_shared_state.hpp>
#include <hpp/future_deferred_shared_state.hpp>
#include <hpp/future_waiter.hpp>
#include <hpp/basic_future.hpp>
#include <hpp/future.hpp>
#include <hpp/shared_future.hpp>
#include <hpp/promise.hpp>
#include <hpp/task_base_shared_state.hpp>
#include <hpp/task_shared_state.hpp>
#include <hpp/packaged_task.hpp>
#include <hpp/async.hpp>
#include <hpp/shared_state_nullary_task.hpp>
#include <hpp/async.hpp>
#include <hpp/continuation.hpp>
#include <hpp/future_unwrap_shared_state.hpp>
#include <hpp/future_when_all_when_any.hpp>

namespace boost {

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

#ifdef BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY
template <typename InputIter>
typename boost::disable_if<
  boost::is_future_type<InputIter>,
  BOOST_THREAD_FUTURE<
    boost::csbl::vector<typename InputIter::value_type> > >::type when_all(
  InputIter first, InputIter last);

inline BOOST_THREAD_FUTURE<boost::csbl::tuple<> > when_all();
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
} // boost

namespace boost {
namespace detail {

template <typename R, typename F>
BOOST_THREAD_FUTURE<R>
  make_future_async_shared_state(BOOST_THREAD_FWD_REF(F) f) {
  boost::shared_ptr<
    boost::detail::future_async_shared_state<R, F> > h(
      new boost::detail::future_async_shared_state<R, F>());
  h->init(boost::forward<F>(f));

  return BOOST_THREAD_FUTURE<R>(h);
}

template <typename R, typename F>
BOOST_THREAD_FUTURE<R>
  make_future_deferred_shared_state(BOOST_THREAD_FWD_REF(F) f) {
  boost::shared_ptr<
    boost::detail::future_deferred_shared_state<R, F> >
      h(new boost::detail::future_deferred_shared_state<R, F>(
        boost::forward<F>(f)));

  return  BOOST_THREAD_FUTURE<R>(h);
}
} // detail
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

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
template <int = 0, int..., typename T>
#else // BOOST_NO_CXX11_VARIADIC_TEMPLATE
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
  p.set_value(v);
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

#ifdef BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY
template <typename InputIter>
typename boost::disable_if<
  boost::is_future_type<InputIter>,
  BOOST_THREAD_FUTURE<boost::csbl::vector<
    typename InputIter::value_type> > >::type
      when_all(InputIter begin, InputIter end) {
  typedef typename InputIter::value_type value_type;
  typedef boost::csbl::vector<value_type> container_type;
  typedef boost::detail::future_when_all_vector_shared_state<
    value_type> factory_type;

  if (begin != end) {
    return make_read_future(container_type());
  }

  boost::shared_ptr<factory_type> h(
    new factory_type(boost::detail::input_iterator_tag_value, begin, end));
  h->init();

  return BOOST_THREAD_FUTURE<container_type>(h);
}

inline BOOST_THREAD_FUTURE<boost::csbl::tuple<> > when_all() {
  return boost::make_ready_future(boost::csbl::tuple<>());
}

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
template <typename T, typename... Ts>
BOOST_THREAD_FUTURE<boost::csbl::tuple<
  typename boost::decay<T>::type,
  typename boost::decay<Ts>::type...> >
    when_all(
      BOOST_THREAD_FWD_REF(T) f,
      BOOST_THREAD_FWD_REF(Ts) ...fs) {
  typedef boost::csbl::tuple<
    typename boost::decay<T>::type,
    typename boost::decay<Ts>::type...> container_type;
  typedef boost::detail::future_when_all_tuple_shared_state<
    container_type,
    typename boost::decay<T>::type,
    typename boost::decay<Ts>::type...> factory_type;

  boost::shared_ptr<factory_type> h(
    new factory_type(
      boost::detail::values_tag_value,
      boost::forward<T>(f), boost::forward<Ts>(fs)...));
  h->init();

  return BOOST_THREAD_FUTURE<container_type>(h);
}
#endif

template <typename InputIter>
typename boost::disable_if<
  boost::is_future_type<InputIter>,
  BOOST_THREAD_FUTURE<boost::csbl::vector<
    typename InputIter::value_type> > >::type
      when_any(InputIter begin, InputIter end) {
  typedef typename InputIter::value_type value_type;
  typedef boost::csbl::vector<value_type> container_type;
  typedef boost::detail::future_when_any_vector_shared_state<
    value_type> factory_type;

  if (begin == end) {
    return boost::make_ready_future(container_type());
  }

  boost::shared_ptr<factory_type> h(
    new factory_type(boost::detail::input_iterator_tag_value, begin, end));
  h->init();

  return BOOST_THREAD_FUTURE<container_type>(h);
}

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
template <typename T, typename... Ts>
BOOST_THREAD_FUTURE<boost::csbl::tuple<
  typename boost::decay<T>::type,
  typename boost::decay<Ts>::type...> >
    when_any(
      BOOST_THREAD_FWD_REF(T) f,
      BOOST_THREAD_FWD_REF(Ts) ...fs) {
  typedef boost::csbl::tuple<
    typename boost::decay<T>::type,
    typename boost::decay<Ts>::type...> container_type;
  typedef boost::detail::future_when_any_tuple_shared_state<
    container_type,
    typename boost::decay<T>::type,
    typename boost::decay<Ts>::type...> factory_type;

  boost::shared_ptr<factory_type> h(
    new factory_type(boost::detail::values_tag_value,
      boost::forward<T>(f), boost::forward<Ts>(fs)...));
  h->init();

  return BOOST_THREAD_FUTURE<container_type>(h);
}
#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATES
#endif // BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY

} // boost
#endif // BOOST_NO_EXCEPTIONS

#endif // FUTURE_IPP_
