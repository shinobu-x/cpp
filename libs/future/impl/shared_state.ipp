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
#elif defined BOOST_THREAD_USES_MOVE
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
};
