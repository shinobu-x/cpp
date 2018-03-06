#ifndef PACKAGED_TASK_IPP
#define PACKAGED_TASK_IPP
#include <include/futures.hpp>
namespace boost {

#ifdef BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
#ifdef BOOST_THREAD_PROVIDES_VARIADIC_THREAD
template <typename R, typename... As>
class packaged_task<R(As...)> {
  typedef typename boost::detail::task_base_shared_state<R(As...)> shared_state;
#else // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
template <typename R>
class packaged_task<R()> {
  typedef typename boost::detail::task_base_shared_state<R()> shared_state;
  typedef boost::shared_ptr<
    boost::detail::task_base_shared_state<R()> > task_ptr;
  boost::shared_ptr<
    boost::detail::task_base_shared_state<R()> > task_;
#endif // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
#else // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
template <typename R>
class packaged_task {
  typedef typename boost::detail::task_base_shared_state<R> shared_state;
  typedef boost::shared_ptr<
    boost::detail::task_base_shared_state<R> > task_ptr;
  boost::shared_ptr<
    boost::detail::task_base_shared_state<R> > task_;
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK

  typedef boost::shared_ptr<shared_state> task_ptr;
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

~packaged_task() {
  if (task_) {
    task_->owner_destroyed();
  }
}

packaged_task(
  BOOST_THREAD_RV_REF(packaged_task) that) BOOST_NOEXCEPT :
  future_obtained_(BOOST_THREAD_RV(that).future_obtained_) {
  task_.swap(BOOST_THREAD_RV(that).task_);
  BOOST_THREAD_RV(that).future_obtained_ = false;
}

packaged_task& operator=(
  BOOST_THREAD_RV_REF(packaged_task) that) BOOST_NOEXCEPT {
#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
  packaged_task temp(boost::move(that));
#else // BOOST_NO_CXX11_RVALUE_REFERENCES
  packaged_task temp(
    static_cast<BOOST_THREAD_RV_REVF(packaged_task)>(that));
#endif // BOOST_NO_CXX11_RVALUE_REFERENCES
  swap(temp);

  return *this;
}

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
void set_executor(executor_ptr_type ex) {
  if (!valid()) {
    boost::throw_exception(boost::task_moved());
  }
  boost::lock_guard<boost::mutex> lock(task_->mutex_);
  task_->set_executor_policy(ex, lock);
}
#endif // BOOST_THREAD_PROVIDES_EXECUTORS

void reset() {
  if (!valid()) {
    boost::throw_exception(
      boost::future_error(
        boost::system::make_error_code(
          boost::future_errc::no_state)));
  }
  task_->reset();
  future_obtained_ = false;
}

void swap(packaged_task that) BOOST_NOEXCEPT {
  task_.swap(that.task_);
  std::swap(future_obtained_, that.future_obtained_);
}

bool valid() const BOOST_NOEXCEPT {
  return task_.get() != 0;
}

BOOST_THREAD_FUTURE<R> get_future() {
  if (!task_) {
    boost::throw_exception(boost::task_moved());
  } else if (!future_obtained_) {
    future_obtained_ = true;
    return BOOST_THREAD_FUTURE<R>(task_);
  } else {
    boost::throw_exception(boost::future_already_retrieved());
  }
}

#if defined(BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK) &&                 \
    defined(BOOST_THREAD_PROVIDES_VARIADIC_THREAD)
void operator()(As ...as) {
  if (!task_) {
    boost::throw_exception(boost::task_moved());
  }
  task_->run(boost::move(as)...);
}

void make_ready_at_thread_exit(As ...as) {
  if (!task_) {
    boost::throw_exception(boost::task_moved());
  }
  if (task_->has_value()) {
    boost::throw_exception(boost::promise_already_satisfied());
  }
  task_->apply(boost::move(as)...);
}
#else // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
      // BOOST_THREAD_PROVIDES_VARIADIC_THREAD
void operator()() {
  if (!task_) {
    boost::throw_exception(boost::task_moved());
  }
  task_->run();
}

void make_ready_at_thread_exit() {
  if (!task_) {
    boost::throw_exception(boost::task_moved());
  }
  if (task_->has_value()) {
    boost::throw_exception(boost::promise_already_satisfied());
  }
  task_->apply();
#endif // BOOST_THREAD_PROVIDES_SIGNATURE_PACKAGED_TASK
       // BOOST_THREAD_PROVIDES_VARIADIC_THREAD

template <typename F>
void set_wait_callback(F f) {
  task_->set_wait_callback(f, this);
}
};

BOOST_THREAD_DCL_MOVABLE_BEG(T)
boost::packaged_task<T>
BOOST_THREAD_DCL_MOVABLE_END

} //boost

#endif // PACKAGED_TASK_IPP
