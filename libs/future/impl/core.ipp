#ifndef CORE_IPP
#define CORE_IPP

#include "../include/futures.hpp"

namespace boost {

template <typename T>
boost::shared_ptr<T> static_shared_from_this(T* that) {
  return boost::static_pointer_cast<T>(that->shared_from_this());
}

template <typename T>
boost::shared_ptr<T const> static_shared_from_this(T const* that) {
  return boost::static_pointer_cast<T const>(that->shared_from_this());
}

class executor;

typedef boost::shared_ptr<executor> executor_ptr_type;

template <typename R>
class BOOST_THREAD_FUTURE;

template <typename R>
class shared_future;

template <typename R>
class promise;
template <typename R>
class packaged_task;

template <typename T>
struct is_future_type<BOOST_THREAD_FUTURE<T> > : boost::true_type {};

template <typename T>
struct is_future_type<shared_future<T> > : boost::true_type {};

namespace detail {

struct relocker {

  boost::unique_lock<boost::mutex>& lock_;

  relocker(boost::unique_lock<boost::mutex>& lock) : lock_(lock) {
    lock_.unlock();
  }

  ~relocker() {

    if (!lock_.owns_lock()) {
      lock_.lock();
    }

  }

  void lock() {

    if (!lock_.owns_lock()) {
      lock_.lock();
    }
  }

private:
  relocker& operator=(relocker const&);
};

class base_future {
public:
};

} // detail
} // boost

#endif // CORE_HPP
