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
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c) :
    f_(boost::move(f)),
    c_(boost::move(c)) {}
  ~continuation_shared_state() {}

  void init(boost::unique_lock<boost::mutex>& lock) {
    f_.future_->set_continuation_ptr(this->shared_from_this(), lock);
  }

  void call() {
    try {
      this->mark_finished_with_result(
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

template <typename F, typename C, typename S>
struct continuation_shared_state<F, void, C, S> : S {
  F f_;
  C c_;

  continuation_shared_state(
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_RV_REF(C) c) :
    f_(boost::move(f)),
    c_(boost::move(c)) {}
  ~continuation_shared_state() {}

  void call() {
    try {
      this->c_(boost::move(this->f_));
      this->mark_finish_with_result();
    } catch (...) {
      this->mark_exceptional_finish();
    }
    this->f_ = F();
  }

  void call(boost::unique_lock<boost::mutex>& lock) {
    try {
      {
        relocker relock(lock);
        this->c_(boost::move(this->f_));
        this->f_ = F();
      }
      this->mark_finished_with_result_internal(lock);
    } catch (...) {
      this->mark_exceptional_finish();
    }
  }

  static void run(boost::shared_ptr<boost::detail::shared_state_base> that) {
    continuation_shared_state* that_ =
      static_cast<continuation_shared_state*>(that.get());
    that_->call();
  }
};

template <typename F, typename R, typename C>
struct future_async_continuation_shared_state :
  boost::detail::continuation_shared_state<
    F,
    R,
    C,
    boost::detail::future_async_shared_state_base<R> > {
  typedef boost::detail::continuation_shared_state<
    F,
    R,
    C,
    boost::detail::future_async_shared_state_base<R> > base_type;

  future_async_continuation_shared_state(
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c) :
    base_type(boost::move(f), boost::forward<C>(c)) {}

  void launch_continuation() {
#ifdef BOOST_THREAD_FUTURE_BLOCKING
    boost::lock_guard<boost::mutex> lock(this->mutex_);
    this->thr_ =
      boost::thread(
        &future_async_continuation_shared_state::run,
        static_shared_from_this(this));
#else
    boost::thread(
      &base_type::run,
      static_shared_from_this(this)).detach();
#endif // BOOST_THREAD_FUTURE_BLOCKING
  }
};

template <typename F, typename R, typename C>
struct future_sync_continuation_shared_state :
  boost::detail::continuation_shared_state<
    F,
    R,
    C,
    boost::detail::shared_state<R> > {
  typedef boost::detail::continuation_shared_state<
    F,
    R,
    C,
    boost::detail::shared_state<R> > base_type;

  future_sync_continuation_shared_state(
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c) :
    base_type(boost::move(f), boost::forward(c)) {}

  void launch_continuation() {
    this->call();
  }
};

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
template <typename S>
struct run_it {
  boost::shared_ptr<S> ex_;

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
  BOOST_THREAD_COPYABLE_AND_MOVABLE(run_it)

  run_it(run_it const& ex) : ex_(ex) {}
  run_it& operator=(BOOST_THREAD_COPY_ASSIGN_REF(run_it) ex) {
    if (this != &ex) {
      ex_ = ex.ex_;
      ex.ex_.reset();
    }
    return *this;
  }
#endif // BOOST_NO_CXX11_RVALUE_REFERENCES
  run_it(boost::shared_ptr<S> ex) : ex_(ex) {}
  void operator()() {
    ex_->run(ex_);
  }
};

} // detail

BOOST_THREAD_DCL_MOVABLE_BEG(F)
boost::detail::run_it<F>
BOOST_THREAD_DCL_MOVABLE_END

namespace detail {

template <typename F, typename R, typename C>
struct future_executor_continuation_shared_state :
  boost::detail::continuation_shared_state<F, R, C> {
  typedef boost::detail::continuation_shared_state<F, R, C> base_type;

  future_executor_continuation_shared_state(
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c) :
    base_type(boost::move(f), boost::forward<C>(c)) {}
  ~future_executor_continuation_shared_state() {}

  template <typename Ex>
  void init(boost::unique_lock<boost::mutex>& lock, Ex ex) {
    this->set_executor_policy(boost::executor_ptr_type(
      new executor_ref<Ex>(ex)), lock);
    this->base_type::init(lock);
  }

  void launch_continuation() {
    boost::detail::run_it<base_type> f(static_shared_from_this(this));
    this->get_executor()->submit(boost::move(f));
  }
};
#endif // BOOST_THREAD_PROVIDES_EXECUTORS
} // detail
} // boost

#endif // BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
#endif // CONTINUATION_IPP
