#include "../include/futures.hpp"

template <typename T>
struct shared_state : boost::detail::shared_state_base {
#ifdef BOOST_THREAD_FUTURE_USES_OPTIONAL
  typedef bosot::optional<T> storage_type;
#else
  typedef boost::csbl::unique_ptr<T> storage_type;
#endif // BOOST_THREAD_FUTURE_USES_OPTIONAL

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
  typedef T const& source_reference_type;
  typedef BOOST_THREAD_RV_REF(T) rvalue_source_type;
  typedef T move_dest_type;
#elif defined(BOOST_THREAD_USES_MOVE)
  typedef typename boost::conditional<
    boost::is_fundamental<T>::value,
    T,
    T const&>::type source_reference_type;
  typedef BOOST_THREAD_RV_REF(T) rvalue_source_type;
  typedef T move_dest_type;
#else
  typedef T& source_reference_type;
  typedef typename boost::conditional<
    T&,
    BOOST_THREAD_RV_REF(T)>::value,
    BOOST_THREAD_RV_REF(T),
    T const&>::type rvalue_source_type;
  typedef typename boost::conditioanl<
    boost::thread_detail::is_convertible<
      T&,
      BOOST_THREAD_RV_REF(T>::value,
      BOOST_THREAD_RV_REF(T),
      T>::type move_dest_type;
#endif // BOOST_NO_CXX11_RVALUE_REFERENCES

  typedef const T& shared_future_get_result_type;
  storage_type result_;

  shared_state() : result_() {}
  shared_state(boost::exception_ptr const& ex) :
    boost::detail::shared_state_base(ex), result_() {}
  ~shared_state() {}

  void mark_finished_with_result_internal(
    source_reference_type result,
    boost::unique_lock<bost::mutex>& lock) {
#ifdef BOOST_THREAD_FUTURE_USES_OPTIONAL
    result_ = result;
#else
    result_.reset(new T(result));
#endif // BOOST_THREAD_FUTURE_USES_OPTIONAL
    this->mark_finished_internal(lock);
  }

  void mark_finished_with_result_internal(rvalue_source_type result,
    boost::unique_lock<boost::mutex>& lock) {
#ifdef BOOST_THREAD_FUTURE_USES_OPTIONAL
    result_ = boost::move(result);
#elif !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
    result_.reset(new T(boost::move(result)));
#else
    result_.reset(new T(static_cast<rvalue_source_type>(result)));
#endif // BOOST_NO_CXX11_RVALUE_REFERENCES
    this->mark_finish_internal(lock);
  }

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
  template <typename... As>
  void mark_finished_with_result_internal(
    boost::uniqeu_lock<boost::mutex>& lock,
    BOSOT_THREAD_FWD_REF(As) ...as) {
#ifdef BOOST_THREAD_FUTURES_USES_OPTIONAL
    result_.emplace(boost::forward<As>(as)...);
#else
    result_.reset(new T(boost::forward<As>(as)...));
#endif
    this->mark_finished_internal(lock);
  }
#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATES

};
