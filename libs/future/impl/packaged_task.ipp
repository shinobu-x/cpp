#ifndef PACKAGED_TASK_IPP
#define PACKAGED_TASK_IPP

#include "../include/futures.hpp"

namespace boost {

#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
template <typename R, typename... Ts>
class packaged_task<R(Ts...)> {
  typedef boost::shared_ptr<
    boost::detail::task_base_shared_state<R(Ts...)> > task_ptr;
#else
template <typename R>
class packaged_task<R()> {
  typedef boost::shared_ptr<
    boost::detail::task_base_shared_state<R()> > task_ptr;
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
#else // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
template <typename R>
class packaged_task {
  typedef boost::shared_ptr<
    boost::detail::task_base_shared_state<R> > task_ptr;
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK

  task_ptr task_;
  bool future_obtained_;
  struct dummy;

public:
  typedef R result_type;
  BOOST_THREAD_MOVABLE_ONLY(packaged_task)
  packaged_task() : future_obtained_(false) {}

#ifdef BOOST_THREAD_RVALUE_REFERENCES_DONT_MATCH_FUNCTION_PTR
#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
explicit packaged_task(
  R(*f)(),
  BOOST_THREAD_FWD_REF(As)... as) {
  typedef R(*FR)(BOOST_THREAD_FWD_REF(As)...);
  typedef boost::detail::task_shared_state<
    FR, R(As...)> task_shared_state_type;

  task_ = task_ptr(
    new task_shared_state_type(f, boost::move(as)...));
}

} //boost

#endif // PACKAGED_TASK_IPP
