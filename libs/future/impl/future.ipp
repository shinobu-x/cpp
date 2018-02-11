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
  template <typename R, typename C, typename Ex>
  friend BOOST_THREAD_FUTURE<R>
    boost::detail::make_future_executor_shared_state(
      Ex& ex,
      BOOST_THREAD_FWD_REF(C) c);

  template <typename Ex, typename F, typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
    boost::detail::make_future_executor_continuation_shared_state(
      Ex& e,
      boost::unique_lock<boost::mutex>& lock,
      F f,
      BOOST_THREAD_FWD_REF(C) c);

  template <typename Ex, typename F, typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
    boost::detail::make_shared_future_executor_continuation_shared_state(
      Ex& e,
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
  BOOST_THREAD_FUTURE(future_ptr future) : base_type(future) {}

public:
  BOOST_THREAD_MOVABLE_ONLY(BOOST_THREAD_FUTURE)
  typedef boost::future_state::state state;
  typedef T value_type;

  // Constructor
  BOOST_CONSTEXPR BOOST_THREAD_FUTURE() {}
  BOOST_THREAD_FUTURE(boost::exceptional_ptr const& e) : base_type(e) {}

  // Destructor
  ~BOOST_THREAD_FUTURE() {}

  // Copy constructor and assignment
  BOOST_THREAD_FUTURE(
    BOOST_THREAD_RV_REF(BOOST_THREAD_FUTURE) that) BOOST_NOEXCEPT :
      base_type(
        boost::move(
          static_cast<base_type&>(BOOST_THREAD_RV(that)))) {}

#ifdef BOOST_PROVIDES_FUTURE_UNWRAP
  inline explicit BOOST_THREAD_FUTURE(
    BOOST_THREAD_RV_REF(
      BOOST_THREAD_FUTURE<BOOST_THREAD_FUTURE<R> >) that);
#endif // BOOST_PROVIDES_FUTURE_UNWRAP

  explicit BOOST_THREAD_FUTURE(
    BOOST_THREAD_RV_REF(shared_future<T>) that) :
      base_type(
        boost::move(
          static_cast<base_type&>(BOOST_THREAD_RV(that)))) {}

  BOOST_THREAD_FUTURE& operator=(
    BOOST_THREAD_RV_REF(
      BOOST_THREAD_FUTURE) that) BOOST_NOEXCEPT {
    this->base_type::operator=(
      boost::move(
        static_cast<base_type&>(BOOST_THREAD_RV(that))));
    return *this;
  }

  shared_future<T> share() {

    return shared_future<T>(boost::move(*this));

  }

  void set_async() {

    this->future_->set_async();

  }

  void set_deferred() {

    this->future_->set_deferred();

  }

  bool run_if_is_deferred() {

    this->future_->run_if_is_deferred();

  }

  bool run_if_is_deferred_or_ready() {

    this->future_->run_if_is_deferred_or_ready();

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
    move_dest_type>::type
      get_or(BOOST_THREAD_RV_REF(T2) v) {

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

  template <typename T2>
  typename boost::disable_if<
    boost::is_void<T2>,
    move_dest_type>::type
      get_or(T2 const& v) {

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
    if (this->future_->has_valid(lock)) {
      return this->future_->get(lock);
    } else {
      return v;
    }
  }

#ifdef BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
  template <typename F>
  inline BOOST_THREAD_FUTURE<
    typename boost::result_of<
      F(BOOST_THREAD_FUTURE)>::type>
        then(BOOST_THREAD_FWD_REF(F) f);

  template <typename F>
  inline BOOST_THREAD_FUTURE<
    typename boost::result_of<
      F(BOOST_THREAD_FUTURE)>::type>
        then(
          boost::launch policy,
          BOOST_THREAD_FWD_REF(F) f);

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
  template <typename Ex, typename F>
  inline BOOST_THREAD_FUTURE<
    typename boost::result_of<
      F(BOOST_THREAD_FUTURE)>::type>
        then(
          Ex& ex,
          BOOST_THREAD_FWD_REF(F) f);
#endif // BOOST_THREAD_PROVIDES_EXECUTORS

  template <typename T2>
  inline typename boost::disable_if<
    boost::is_void<T2>,
    BOOST_THREAD_FUTURE<T> >::type
      fallback_to(BOOST_THREAD_RV_REF(T2) v);

  template <typename T2>
  inline typename boost::disable_if<
    boost::is_void<T2>,
    BOOST_THREAD_FUTURE<T> >::type
      fallback_to(T2 const& v);
#endif // BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
};

BOOST_THREAD_DCL_MOVABLE_BEG(T)
BOOST_THREAD_FUTURE<T>
BOOST_THREAD_DCL_MOVABLE_END

template <typename T2>
class BOOST_THREAD_FUTURE<boost::BOOST_THREAD_FUTURE<T2> > :
  public boost::detail::basic_future<boost::BOOST_THREAD_FUTURE<T2> > {
  typedef boost::BOOST_THREAD_FUTURE<T2> T;
  typedef boost::detail::basic_future<T2> base_type;
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
    boost::detail::future_async_continuation_shared_state(
      boost::unique_lock<boost::mutex>& lock,
      BOOST_THREAD_RV_REF(F) f,
      BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
    boost::detail::future_sync_continuation_shared_state(
      boost::unique_lock<boost::mutex>& lock,
      BOOST_THREAD_RV_REF(F) f,
      BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
    boost::detail::future_deferred_continuation_shared_state(
      boost::unique_lock<boost::mutex>& lock,
      BOOST_THREAD_RV_REF(F) f,
      BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
    boost::detail::shared_future_async_continuation_shared_state(
      boost::unique_lock<boost::mutex>& lock,
      F f,
      BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
    boost::detail::shared_future_sync_continuation_shared_state(
      boost::unique_lock<boost::mutex>& lock,
      F f,
      BOOST_THREAD_FWD_REF(C) c);

  template <typename F, typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
    boost::detail::shared_future_deferred_continuation_shared_state(
      boost::unique_lock<boost::mutex>& lock,
      F f,
      BOOST_THREAD_FWD_REF(C) c);

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
  template <typename R, typename C, typename Ex>
  friend BOOST_THREAD_FUTURE<R>
    boost::detail::make_future_executor_continuation_shared_state(
      Ex& ex,
      BOOST_THREAD_FWD_REF(C) c);

  template <typename Ex, typename F, typename R, typename C>
  friend BOOST_THREAD_FUTURE<R>
    boost::detail::make_shared_future_executor_continuation_shared_state(
      Ex& ex,
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
      boost::unique_lock<boost;:lock>& lock,
      BOOST_THREAD_RV_REF(F) f);
#endif // BOOST_THREAD_PROVIDES_FUTURE_UNWRAP

#ifdef BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY
  template <typename InputIter>
  friend typename boost::disable_if<
    boost::is_future_type<InputIter>,
    BOOST_THREAD_FUTURE<
      boost::csbl::vector<
        typename InputIter::value_type> > >::type
          when_all(InputIter begin, InputIter end);

  inline BOOST_THREAD_FUTURE<boost::csbl::tuple<> > when_all();

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
  template <typename T, typename... Ts>
  friend BOOST_THREAD_FUTURE<
    boost::csbl::tuple<
      typename boost::decay<T>::type,
      typename boost::decay<Ts>::type...> >
        when_all(
          BOOST_THREAD_FWD_REF(T) F,
          BOOST_THREAD_FWD_REF(Ts) ...fs);
#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATES

  template <typename InputIter>
  friend typename boost::disable_if<
    boost::is_future_type<InputIter>,
    BOOST_THREAD_FUTURE<
      boost::csbl::vector<
        typename InputIter::value_type> > >::type
          when_any(InputIter begin, InputIter end);

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
  template <typename T, typename Ts...>
  friend BOOST_THREAD_FUTURE<
    boost::csbl::tuple<
      typename boost::decay<T>::type,
      typename boost::decay<Ts>::type...> >
        when_any(
          BOOST_THREAD_FWD_REF(T) f,
          BOOST_THREAD_FWD_REF(Ts) ...fs);
#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATES
#endif // BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY

} // boost

#endif // FUTURE_IPP
