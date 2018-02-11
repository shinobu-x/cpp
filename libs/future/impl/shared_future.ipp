#ifndef SHARED_FUTURE_IPP
#define SHARED_FUTURE_IPP

#include "../include/futures.hpp"

namespace boost {

template <typename T>
class shared_future : public boost::detail::basic_future<T> {
  typedef boost::detail::basic_future<T> base_type;
  typedef typename base_type::future_ptr future_ptr;

  friend class boost::detail::future_waiter;
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
#endif // BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION

#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
  template <class>
  friend class packaged_task;
#else
  friend class packaged_task<T>;
#endif

  // Constructor
  shared_future(future_ptr future) : base_type(future) {}

public:
  BOOST_THREAD_COPYABLE_AND_MOVABLE(shared_future);
  typedef T value_type;
  typedef boost::future_state::state state;

  // Constructor
  BOOST_CONSTEXPR shared_future() {}

  // Destructor
  ~shared_future() {}

  shared_future(shared_future const& that) : base_type(that.future_) {}

  shared_future(boost::exceptional_ptr const& e) : base_type(e) {}

  shared_future(
    BOOST_THREAD_RV_REF(shared_future) that) BOOST_NOEXCEPT :
    base_type(
      boost::move(
        static_cast<base_type&>(
          BOOST_THREAD_RV(that)))) {}

  shared_future(
    BOOST_THREAD_RV_REF(BOOST_THREAD_FUTURE<T>) that) BOOST_NOEXCEPT :
    base_type(
      boost::move(
        static_cast<base_type&>(
          BOOST_THREAD_RV(that)))) {}

  shared_future& operator=(
    BOOST_THREAD_COPY_ASSIGN_REF(shared_future) that) {
    this->future_ = that.future_;
    return *this;
  }

  shared_future& operator=(
    BOOST_THREAD_RV_REF(shared_future) that) BOOST_NOEXCEPT {
    base_type::operator=(
      boost::move(
        static_cast<base_type&>(
          BOOST_THREAD_RV(that))));
    return *this
  }

  shared_future& operator=(
    BOOST_THREAD_RV_REF(BOOST_THREAD_FUTURE<T>) that) BOOST_NOEXCEPT {
    base_type::operator=(
      boost::move(
        static_cast<base_type&>(
          BOOST_THREAD_RV(that))));
    return *this;
  }


} // boost

#endif // SHARED_FUTURE_IPP
