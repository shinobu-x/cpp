#ifndef TASK_SHARED_STATE_IPP
#define TASK_SHARED_STATE_IPP

#include "../include/futures.hpp"

namespace boost {
namespace detail {

#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
template <typename F, typename R>
struct task_shared_state;
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
template <typename F, typename R, typename... As>
struct task_shared_state<F, R(As...)> :
  boost::detail::task_base_shared_state<R(As...)> {
#else // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
template <typename F, typename R>
struct task_shared_state<F, R()> :
  boost::detail::task_base_shared_state<R()> {
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
#else
template <typename F, typename R>
struct task_shared_state :
  boost::detail::task_base_shared_state<R> {
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
private:
  task_shared_state(task_shared_state&);

public:
  F f_;
  task_shared_state(F const& f) : f_(f) {}

  task_shared_state(BOOST_THREAD_RV_REF(F) f) : f_(boost::move(f)) {}

  F callable() {
    return boost::move(f_);
  }
#if defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK) &&                  \
    defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
  void do_apply(BOOST_THREAD_RV_REF(As) ...as) {
    try {
      this->set_value_at_thread_exit(boost::move(as)...);
#else
  void do_apply() {
    try {
      this->set_value_at_thread_exit(f_());
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
       // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
    } catch (...) {
      this->set_exception_at_thread_exit(boost::current_exception());
    }
  }

#if defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK) &&                  \
    defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
  void do_run(BOOST_THREAD_RV_REF(As) ...as) {
    try {
      this->mark_finished_with_result(f_(boost::move(as)...));
#else // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
      // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
  void do_run() {
    try {
#if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
      R r((f_()));
      this->mark_finished_with_result(boost::move(r));
#else // BOOST_NO_CXX11_RVALUE_REFERENCES
      this->mark_finished_with_result(f_());
#endif // BOOST_NO_CXX11_RVALUE_REFERENCES
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
       // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
    } catch (...) {
      this->mark_exceptional_finish();
    }
  }
};

} // detail
} // boost
 
#endif // TASK_SHARED_STATE_IPP
