#ifndef CONTINUATION_IPP
#define CONTINUATION_IPP

#include <include/futures.hpp>

namespace boost {
#ifdef BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
namespace detail {

template <
  typename F,
  typename R,
  typename C,
  typename S = boost::detail::shared_state<R> >
struct continuation_shared_state : S {
  F f_;
  C c_;

  continuation_shared_state(
    BOOST_THREAD_RV_REF f,
    BOOST_THREAD_FWD_REF(C) c) :
    f_(boost::move(f)),
    c_(boost::move(c)) {}

  ~continuation_shared_state() {}

  void init(boost::unique_lock<boost::mutex>& lock) {
    f_.future_->set_continuation_ptr(this->shared_from_this(), lock);
  }

  void call() {
    try {
      this->mark_finish_with_result(
        this->c_(boost::move(this->f_)));
    } catch (...) {
      this->mark_exceptional_finish();
    }
  }

  void call(boost::unique_lock<boost::mutex>& lock) {
    try {
      relocker relock(lock);
      R r = this->c_(boost::move(this->f_));
      this->f_ = F();
      relock.lock();
      this->mark_finished_with_result_internal(boost::move(r), lock);
    } catch (...) {
      this->mark_exceptional_finish_internal(
        boost::current_exception(), lock);
      relocker relock(lock);
      this->f_ = F();
    }
  }

  static void run(boost::shared_ptr<boost::detail::shared_state_base> that) {
    continuation_shared_state* that_ =
      static_cast<continuation_shared_state*>(that.get());
    that_->call();
  }
};

} // detail
} // boost
#endif // CONTINUATION_IPP
