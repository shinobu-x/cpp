#ifndef CORE_HPP
#define CORE_HPP

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

} // detail
} // boost

#endif // CORE_HPP
