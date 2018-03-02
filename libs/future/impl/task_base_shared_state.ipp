#ifndef TASK_BASE_SHARED_STATE_IPP
#define TASK_BASE_SHARED_STATE_IPP
#include <include/futures.hpp>

namespace boost {
namespace detail {

#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
template <typename R>
struct task_base_shared_state;
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
template <typename R, typename... As>
struct task_base_shared_state<R(As...)> :
#else
template <typename R>
struct task_base_shared_state<R()> :
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
#else // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
template <typename R>
struct task_base_shared_state :
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
  boost::detail::shared_state<R> {
  bool started_;

  task_base_shared_state() : started_(false) {}

  void reset() {
    started_ = false;
    this->validate();
  }

#if defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK) &&                  \
    defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
  virtual void do_run(BOOST_THREAD_RV_REF(As) ...as) = 0;
  void run(BOOST_THREAD_RV_REF(As) ...as) {
#else
  virtual void do_run() = 0;
  void run() {
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
       // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
    {
      boost::lock_guard<boost::mutex> lock(this->mutex_);
      if (started_) {
        boost::throw_exception(boost::task_already_started());
      }
      started_ = true;
    }
#if defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK) &&                  \
    defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
    do_run(boost::move(as)...);
#else
    do_run();
#endif
  }

#if defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK) &&                  \
    defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
  virtual void do_apply(BOOST_THREAD_RV_REF(As) ...as) = 0;
  void apply(BOOST_THREAD_RV_REF(As) ...as) {
#else
  virtual void do_apply() = 0;
  void apply() {
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
       // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
    {
      boost::lock_guard<boost::mutex> lock(this->mutex_);
      if (started_) {
        boost::throw_exception(boost::task_already_started());
      }
      started_ = true;
    }
#if defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK) &&                  \
    defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
    do_apply(boost::move(as)...);
#else
    do_apply();
#endif
  }

  void owner_destroyed() {
    boost::unique_lock<boost::mutex> lock(this->mutex_);
    if (!started_) {
      started_ = true;
      this->mark_exceptional_finish_internal(
        boost::copy_exception(boost::broken_promise()), lock);
    }
  }
};

} // detail
} // boost
#endif // TASK_BASE_SHARED_STATE_IPP
