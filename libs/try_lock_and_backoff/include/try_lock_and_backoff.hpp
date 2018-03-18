#ifndef TRY_LOCK_AND_BACKOFF_HPP
#define TRY_LOCK_AND_BACKOFF_HPP

#include <boost/thread/lock_types.hpp>
#include <boost/thread/mutex.hpp>

template <typename mutex_type, typename... mutex_types>
unsigned try_lock_and_backoff(mutex_type m, mutex_types ...ms){
  unsigned r;

  return r;
}

#endif
