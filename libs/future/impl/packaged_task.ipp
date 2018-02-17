#ifndef PACKAGED_TASK_IPP
#define PACKAGED_TASK_IPP

#include "../include/futures.hpp"

namespace boost {

#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
template <typename R, typename... As>
class packaged_task<R(As...)> {
  typedef boost::shared_ptr<
    boost::detail::task_base_shared_state<R(As...)> > task_ptr;
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
    FR,
    R(As...)> task_shared_state_type;

  task_ = task_ptr(
    new task_shared_state_type(f, boost::move(as)...));
  future_obtained_ = false;
}
#else // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
explicit packaged_task(R(*f)()) {
  typedef R(*FR)();
  typedef boost::detail::task_shared_state<
    FR,
    R()> task_shared_state_type;

  task_ = task_ptr(
    new task_shared_state_type(f));
  future_obtained_ = false;
}
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
#else
explicit packaged_task(R(*f)() {
  typedef R(*FR)();
  typedef boost::detail::task_shared_state<
    FR,
    R> task_shared_state_type;

  task_ = task_ptr(
    new task_shared_state_type(f));
  future_obtain_ = false;
}
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
#endif // BOOST_THREAD_PROVIDES_REFERENCES_DONT_MATCH_FUNCTION_PTR

#ifdef BOOST_NO_CXX11_RVALUE_REFERENCES
template <typename F>
explicit packaged_task(
  BOOST_THREAD_FWD_REF(F) f,
  typename boost::disable_if<
    boost::is_same<
      typename  boost::decay<F>::type,
      packaged_task>,
  dummy* >::type = 0) {
  typedef typename boost::decay<F>::type FR;

#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
  typedef boost::detail::task_shared_state<
    FR,
    R(As...)> task_shared_state_type;
#else // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
  typedef boost::detail::task_shared_state<
    FR,
    R()> task_shared_state_type;
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
#else // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
  typedef boost::detail::task_shared_state<
    FR,
    R> task_shared_state_type;
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK

  task_ = task_ptr(
    new task_shared_state_type(
      boost::forward<F>(f)));
  future_obtained_ = false
}
#else // BOOST_THREAD_PROVIDE_VARIADIC_THREAD
template <typename F>
explicit packaged_task(
  F const& f,
  typename boost::disable_if<
    boost::is_same<
      typename boost::decay<F>::type,
      packaged_task>,
  dummy* >::type = 0) {
#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
  typedef boost::detail::task_shared_state<
    F,
    R(As...)> task_shared_state_type;
#else // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
  typedef boost::detail::task_shared_state<
    F,
    R()> task_shared_state_type;
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
#else
  typedef boost::detail::task_shared_state<
    F,
    R> task_shared_state_type;
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK

  task_ = task_ptr(
    new task_shared_state_type(f));
  future_obtained_ = false;
}

template <typename F>
explicit packaged_task(
  BOOST_THREAD_RV_REF(F) f) {
#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
  typedef boost::detail::task_shared_state<
    F,
    R(As...)> task_shared_state_type;
#else // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
  typedef boost::detail::task_shared_state<
    F,
    R()> task_shared_state_type;
#endif // BOOST_THREAD_PROVIDES_VAIRADIC_THREAD
#else // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
  typedef boost::detail::task_shared_state<
    F,
    R> task_shared_state_type;
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK

  task_ = task_ptr(
    new task_shared_state_type(
      boost::move(f)));
  future_obtained_ = false;
}
#endif // BOOST_NO_CXX11_RVALUE_REFERENCES

#ifdef BOOST_THREAD_PROVIDES_FUTURE_CTOR_ALLOCATOR
#ifdef BOOST_THREAD_RVALUE_REFERENCES_DONT_MATCH_FUNCTION_PTR
template <typename Allocator>
packaged_task(
  boost::allocator_tag_t,
  Allocator alloc,
  R(*f)()) {
  typedef R(*FR)();
#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
  typedef boost::detail::task_shared_state<
    FR,
    R(As...)> task_shared_state_type;
#else // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
  typedef boost::detail::task_shared_state<
    FR,
    R()> task_shared_state_type;
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
#else
  typedef boost::detail::task_shared_state<
    FR,
    R> task_shared_state_type;
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK

  typedef typename Allocator::template rebind<
    task_shared_state_type>::other Alloc;
  typedef boost::thread_detail::allocator_destructor<Alloc> Dtor;

  Alloc alloc_(alloc);
  task_ = task_ptr(
    new(alloc_.allocate(1)) task_shared_state_type(f),
    Dtor(alloc_, 1));
  future_obtained_ = false;

#endif // BOOST_THREAD_RVALUE_REFERENCES_DONT_MATCH_FUNCTION_PTR
#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
template <typename F, typename Allocator>
packaged_task(
  boost::allocator_tag_t,
  Allocator alloc,
  BOOST_THREAD_FWD_REF(F) f) {
  typedef typename boost::decay<F>::type FR;

#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
  typedef boost::detail::task_shared_state<
    FR,
    R(As...)> task_shared_state_type;
#else // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
  typedef boost::detail::task_shared_state<
    FR,
    R()> task_shared_state_type;
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
#else // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
  typedef boost::detail::task_shared_state<
    FR,
    R> task_shared_state_type;
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK

  typedef typename Allocator::template rebind<
    task_shared_state_type>::other Alloc;
  typedef boost::thread_detail::allocator_destructor<Alloc> Dtor;

  Alloc alloc_(alloc);
  task_ = task_ptr(
    new(alloc_.allocate(1)) task_shared_state_type(
      boost::forward<F>(f)),
      Dtor(alloc_, 1));
  future_obtained_ = false;
}
#else // BOOST_NO_CXX11_RVALUE_REFERENCES
template <typename F, typename Allocator>
packaged_task(
  boost::allocator_tag_t,
  const F& f) {
#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
  typedef boost::detail::task_shared_state<
    F,
    R(As...)> task_shared_state_type;
#else // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
  typedef boost::detail::task_shared_state<
    F,
    R()> task_shared_state_type;
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
#else // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
  typedef boost::detail::task_shared_state<
    F,
    R> task_shared_state_type;
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK

  typedef typename Allocator::template rebind<
    task_shared_state_type>::other Alloc;
  typedef boost::thread_detail::allocator_destructor<Alloc> Dtor;

  Alloc alloc_(alloc);
  task_ = task_ptr(
    new(alloc_.allocate(1)) task_shared_state_type(
      boost::move(f)),
    Dtor(alloc_, 1));
  future_obtained_ = false;
}
#endif // BOOST_NO_CXX11_RVALUE_REFERENCES
#endif // BOOST_THREAD_PROVIDES_FUTURE_CTOR_ALLOCATORS
};
} //boost

#endif // PACKAGED_TASK_IPP
