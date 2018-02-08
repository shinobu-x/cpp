#ifndef FUTURE_ASYNC_SHARED_STATE_BASE_IPP
#define FUTURE_ASYNC_SHARED_STATE_BASE_IPP
#include "../include/futures.hpp"

namespace boost {
namespace detail {

template <typename S>
struct future_async_shared_state_base :
  boost::detail::shared_state<S> {
  typedef boost::detail::shared_state<S> base_type;
 
protected:
#ifdef BOOST_THREAD_FUTURE_BLOCKING
  boost::thread thr_;
  void join() {
    if (boost::this_thread::get_id() == thr_.get_id()) {
      thr_.detach();
      return;
    }
 
    if (thr_.joinable()) {
      thr_.join();
    }
  }
#endif // BOOST_THREAD_FUTURE_BLOCKING
public:
  future_async_shared_state_base() {
    this->set_async();
  }
 
  ~future_async_shared_state_base() {
#ifdef BOOST_THREAD_FUTURE_BLOCKING
    join();
#endif // BOOST_THREAD_FUTURE_BLOCKING
  }
 
  virtual void wait(boost::unique_lock<boost::mutex>& lock, bool rethrow) {
#ifdef BOOST_THREAD_FUTURE_BLOCKING
    {
      relocker relock(lock);
      join();
    }
#endif // BOOST_THREAD_FUTURE_BLOCKING
    this->base_type::wait(lock, rethrow);
  }
};

} // detail
} // boost
#endif
