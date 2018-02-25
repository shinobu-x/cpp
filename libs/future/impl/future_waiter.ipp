#ifndef FUTURE_WAITER_IPP
#define FUTURE_WAITER_IPP
#include <include/futures.hpp>

namespace boost {
namespace detail {

class future_waiter {
public:
  typedef std::vector<int>::size_type count_type;

private:
  struct registered_waiter {
    typedef typename boost::detail::shared_state_base shared_state_base;
    boost::shared_ptr<shared_state_base> future_;
    shared_state_base::notify_when_ready_handle handle_;
    count_type index_;

    // Constructor
    registered_waiter(
      boost::shared_ptr<shared_state_base> const& future,
      shared_state_base::notify_when_ready_handle handle,
      count_type index) :
      future_(future), handle_(handle), index_(index) {}
  };

  struct all_futures_lock {
#ifdef _MANAGED
    typedef std::ptrdiff_t count_type_portable;
#else
    typedef count_type count_type_portable;
#endif // _MANAGED
    count_type_portable count_;
    boost::scoped_array<boost::unique_lock<boost::mutex> > locks_;

    // Constructor
    all_futures_lock(
      std::vector<registered_waiter>& futures) :
      count_(futures.size()),
      locks_(new boost::unique_lock<boost::mutex>[count_]) {
      for (count_type_portable i = 0; i < count_; ++i) {
        locks_[i] = BOOST_THREAD_MAKE_RV_REF(
          boost::unique_lock<boost::mutex>(futures[i].future_->mutex_));
      }
    }

    void lock() {
      boost::lock(locks_.get(), locks_.get() + count_);
    }

    void unlock() {
      for (count_type_portable i = 0; i < count_; ++i) {
        locks_[i].unlock();
      }
    }
  };

  boost::condition_variable_any cv_;
  std::vector<registered_waiter> futures_;
  count_type future_count_;

public:
  // Constructor
  future_waiter() : future_count_(0) {}

  template <typename F>
  void add(F& f) {

    if (f.future_) {
      registered_waiter waiter_(
        f.future_, f.future_->notify_when_ready(cv_), future_count_);
      try {
        futures_.push_back(waiter_);
      } catch (...) {
        f.future_->unnotify_when_ready(waiter_.handle_);
      }
    }
    ++future_count_;
  }

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATE
  template <typename F, typename... Fs>
  void add(F& f, Fs& ...fs) {
    add(f);
    add(fs...);
  }
#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATE

  count_type wait() {
    all_futures_lock lock(futures_);
    for (;;) {
      for (count_type i = 0; i < futures_.size(); ++i) {
        if (futures_[i].future_->done_) {
          return futures_[i].index_;
        }
      }
      cv_.wait(lock);
    }
  }

  // Destructor
  ~future_waiter() {
    for (count_type i = 0; i < futures_.size(); ++i) {
      futures_[i].future_->unnotify_when_ready(futures_[i].handle_);
    }
  }
};
} // detail
} // boost

#endif // FUTURE_WAITER_IPP
