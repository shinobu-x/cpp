#ifndef SHARED_STATE_IPP
#define SHARED_STATE_IPP

#include "../include/futures.hpp"

template <typename T>
struct shared_state : boost::detail::shared_state_base {
#ifdef BOOST_THREAD_FUTURE_USES_OPTIONAL
  typedef boost::optional<T> storage_type;
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
  typedef T move_
}

#endif // SHARED_STATE_IPP
