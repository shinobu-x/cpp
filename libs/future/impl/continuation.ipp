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

template <typename F, typename R, typename C>
struct shared_future_async_continuation_shared_state :
  boost::detail::continuation_shared_state<
    F,
    R,
    C,
    boost::detail::future_async_shared_state_base<R> > {
  typedef boost::detail::future_async_shared_state_base<R> base_type;

  shared_future_async_continuation_shared_state(
    F f,
    BOOST_THREAD_FWD_REF(C) c) :
    base_type(boost::move(f), boost::forward<C>(c)) {}

  void launch_continuation() {
#ifdef BOOST_THREAD_FUTURE_BLOCKING
    boost::lock_guard<boost::mutex> lock(this->mutex_);
    this->thr_ =
      boost::thread(
        &base_type::run,
        static_shared_from_this(this));
#else
    boost::thread(
      &base_type::run,
      static_shared_from_this(this)).detach();
#endif // BOOST_THREAD_FUTURE_BLOCKING
  }
};

template <typename F, typename R, typename C>
struct shared_future_sync_continuation_shared_state :
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

  shared_future_sync_continuation_shared_state(
    F f,
   BOOST_THREAD_FWD_REF(C) c) :
   base_type(boost::move(f), boost::forward<C>(c)) {}

  void launch_continuation() {
    this->call();
  }
};

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
template <typename F, typename R, typename C>
struct shared_future_executor_continuation_shared_state :
  boost::detail::continuation_shared_state<F, R, C> {
  typedef boost::detail::continuation_shared_state<F, R, C> base_type;

  shared_future_executor_continuation_shared_state(
    F f,
    BOOST_THREAD_FWD_REF(C) c) :
    base_type(boost::move(f), boost::forward<C>(c)) {}
  ~shared_future_executor_continuation_shared_state() {}

  template <typename Ex>
  void init(boost::unique_lock<boost::mutex>& lock, Ex& ex) {
    this->set_executor_policy(
      boost::executor_ptr_type(
        new executor_ref<Ex>(ex)), lock);
  }

  void launch_continuation() {
    boost::detail::run_it<base_type> f(static_shared_from_this(this));
    this->get_executor()->submit(boost::move(f));
  }
};
#endif // BOOST_THREAD_PROVIDES_EXECUTORS

template <typename F, typename R, typename C>
struct future_deferred_continuation_shared_state :
  boost::detail::continuation_shared_state<F, R, C> {
  typedef boost::detail::continuation_shared_state<F, R, C> base_type;

  future_deferred_continuation_shared_state(
    BOOST_THREAD_RV_REF(F) f,
    BOOST_THREAD_FWD_REF(C) c) :
    base_type(boost::move(f), boost::forward<C>(c)) {
    this->set_deferred();
  }

  virtual void execute(boost::unique_lock<boost::mutex>& lock) {
    this->f_.wait();
    this->call(lock);
  }

  virtual void launch_continuation() {}
};

template <typename F, typename R, typename C>
struct shared_future_deferred_continuation_shared_state :
  boost::detail::continuation_shared_state<F, R, C> {
  typedef boost::detail::continuation_shared_state<F, R, C> base_type;

  shared_future_deferred_continuation_shared_state(
    F f,
    BOOST_THREAD_FWD_REF(C) c) :
    base_type(boost::move(f), boost::forward<C>(c)) {}

  virtual void execute(boost::unique_lock<boost::mutex>& lock) {
    this->f_.wait();
    this->call(lock);
  }

  virtual void launch_continuation() {}
};

template <typename F, typename R, typename C>
BOOST_THREAD_FUTURE<R> make_future_async_continuation_shared_state(
  boost::unique_lock<boost::mutex>& lock,
  BOOST_THREAD_RV_REF(F) f,
  BOOST_THREAD_FWD_REF(C) c) {
  typedef typename boost::decay<C>::type callback_type;
  boost::shared_ptr<
    boost::detail::future_async_continuation_shared_state<
      F,
      R,
      callback_type> > h(
        new boost::detail::future_async_continuation_shared_state<
          F,
          R,
          callback_type>(boost::move(f), boost::forward<C>(c)));
  h->init(lock);

  return BOOST_THREAD_FUTURE<R>(h);
}

template <typename F, typename R, typename C>
BOOST_THREAD_FUTURE<R> make_future_sync_continuation_shared_state(
  boost::unique_lock<boost::mutex>& lock,
  BOOST_THREAD_RV_REF(F) f,
  BOOST_THREAD_FWD_REF(C) c) {
  typedef typename boost::decay<C>::type callback_type;
  boost::shared_ptr<
    boost::detail::future_sync_continuation_shared_state<
      F,
      R,
      callback_type> > h(
        new boost::detail::future_sync_continuation_shared_state<
          F,
          R,
          callback_type>(boost::move(f), boost::forward<C>(c)));
  h->init(lock);

  return BOOST_THREAD_FUTURE<R>(h);
}

template <typename F, typename R, typename C>
BOOST_THREAD_FUTURE<R> make_future_deferred_continuation_shared_state(
  boost::unique_lock<boost::mutex>& lock,
  BOOST_THREAD_RV_REF(F) f,
  BOOST_THREAD_FWD_REF(C) c) {
  typedef typename boost::decay<C>::type callback_type;
  boost::shared_ptr<
    boost::detail::future_deferred_continuation_shared_state<
      F,
      R,
      callback_type> > h(
        new boost::detail::future_deferred_continuation_shared_state<
          F,
          R,
          callback_type>(boost::move(f), boost::forward<C>(c)));
  h->init(lock);

  return BOOST_THREAD_FUTURE<R>(h);
}

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
template <typename Ex, typename F, typename R, typename C>
BOOST_THREAD_FUTURE<R> make_future_executor_continuation_shared_state(
  Ex& ex,
  boost::unique_lock<boost::mutex>& lock,
  BOOST_THREAD_RV_REF(F) f,
  BOOST_THREAD_FWD_REF(C) c) {
  typedef typename boost::decay<C>::type callback_type;
  boost::shared_ptr<
    boost::detail::future_executor_continuation_shared_state<
      F,
      R,
      callback_type> > h(
        new boost::detail::future_executor_continuation_shared_state<
          F,
          R,
          callback_type>(boost::move(f), boost::forward<C>(c)));
    h->init(lock, ex);

  return BOOST_THREAD_FUTURE<R>(h);
}
#endif // BOOST_THREAD_PROVIDES_EXECUTORS

template <typename F, typename R, typename C>
BOOST_THREAD_FUTURE<R> make_shared_future_async_continuation_shared_state(
  boost::unique_lock<boost::mutex>& lock,
  F f,
  BOOST_THREAD_FWD_REF(C) c) {
  typedef typename boost::decay<C>::type callback_type;
  boost::shared_ptr<
    boost::detail::shared_future_async_continuation_shared_state<
      F,
      R,
      callback_type> > h(
        new boost::detail::shared_future_async_continuation_shared_state<
          F,
          R,
          callback_type>(f, boost::forward<C>(c)));
  h->init(lock);

  return BOOST_THREAD_FUTURE<R>(h);
}

template <typename F, typename R, typename C>
BOOST_THREAD_FUTURE<R> make_shared_future_sync_continuation_shared_state(
  boost::shared_ptr<boost::mutex>& lock,
  F f,
  BOOST_THREAD_FWD_REF(C) c) {
  typedef typename boost::decay<C>::type callback_type;
  boost::shared_ptr<
    boost::detail::shared_future_sync_continuation_shared_state<
      F,
      R,
      callback_type> > h(
        new boost::detail::shared_future_sync_continuation_shared_state<
          F,
          R,
          callback_type>>(f, boost::forward<C>(c)));
  h->init(lock);

  return BOOST_THREAD_FUTURE<R>(h);
}
} // detail

template <typename R>
template <typename F>
inline BOOST_THREAD_FUTURE<
  typename boost::result_of<
    F(BOOST_THREAD_FUTURE<R>)>::type> BOOST_THREAD_FUTURE<R>::then(
      boost::launch policy,
      BOOST_THREAD_FWD_REF(F) f) {
  typedef typename boost::result_of<
    F(BOOST_THREAD_FUTURE<R>)>::type future_type;

  BOOST_THREAD_ASSERT_PRECONDITION(
    this->future_.get() != nullptr,
    boost::future_initialized());

  boost::shared_ptr<
    boost::detail::shared_state_base> shared_state(this->future_);
  boost::unique_lock<boost::mutex> lock(shared_state->mutex_);

  if (boost::underlying_cast<int>(policy) &&
      (int)boost::launch::async) {
    return BOOST_THREAD_MAKE_RV_REF(
      (
        boost::detail::make_future_async_continuation_shared_state<
          BOOST_THREAD_FUTURE<R>,
          future_type>(
            lock,
            boost::move(*this),
            boost::forward<F>(f)
    )));
  } else if (boost::underlying_cast<int>(policy) &&
      (int)boost::launch::deferred) {
    return BOOST_THREAD_MAKE_RV_REF(
      (
        boost::detail::make_future_deferred_continuation_shared_state<
          BOOST_THREAD_FUTURE<R>,
          future_type>(
            lock,
            boost::move(*this),
            boost::forward<F>(f)
    )));
#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
  } else if (boost::underlying_cast<int>(policy) &&
             (int)boost::launch::executor) {
    assert(this->future_->get_executor());
    typedef boost::executor Ex;
    Ex& ex = *(this->future_->get_executor());

    return BOOST_THREAD_MAKE_RV_REF(
      (
        boost::detail::make_future_executor_continuation_shared_state<
          Ex,
          BOOST_THREAD_FUTURE<R>,
          future_type>(
            ex,
            lock,
            boost::move(*this),
            boost::forward<F>(f)
    )));
#endif // BOOST_THREAD_PROVIDES_EXECUTORS
  } else if (boost::underlying_cast<int>(policy) &&
             (int)boost::launch::inherit) {
    boost::launch policy_ = this->launch_policy(lock);

    if (boost::underlying_cast<int>(policy) &&
        (int)boost::launch::async) {
      return BOOST_THREAD_MAKE_RV_REF(
        (
          boost::detail::make_future_async_continuation_shared_state<
            BOOST_THREAD_FUTURE<R>,
            future_type>(
              lock,
              boost::move(*this),
              boost::forward<F>(f)
      )));
    } else if (boost::underlying_cast<int>(policy) &&
               (int)boost::launch::deferred) {
      return BOOST_THREAD_MAKE_RV_REF(
        (
          boost::detail::make_future_deferred_continuation_shared_state<
            BOOST_THREAD_FUTURE<R>,
            future_type>(
              lock,
              boost::move(*this),
              boost::forward<F>(f)
      )));
#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
    } else if (boost::underlying_cast<int>(policy) &&
               (int)boost::launch::executor) {
      assert(this->future_->get_executor());
      typedef boost::executor Ex;
      Ex& ex = *(this->future_->get_executor());

      return BOOST_THREAD_MAKE_RV_REF(
        (
          boost::detail::make_future_executor_continuation_shared_state<
            BOOST_THREAD_FUTURE<R>,
            future_type>(
              lock,
              boost::move(*this),
              boost::forward<F>(f)
      )));
#endif // BOOST_THREAD_PROVIDES_EXECUTORS
    } else {
      return BOOST_THREAD_MAKE_RV_REF(
        (
          boost::detail::make_future_async_continuation_shared_state<
            BOOST_THREAD_FUTURE<R>,
            future_type>(
              lock,
              boost::move(*this),
              boost::forward<F>(f)
      )));
    }
  } else {
    return BOOST_THREAD_MAKE_RV_REF(
      (
        boost::detail::make_future_async_continuation_shared_state<
          BOOST_THREAD_FUTURE<R>,
          future_type>(
            lock,
            boost::move(*this),
            boost::forward<F>(f)
    )));
  }
}

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
template <typename R>
template <typename Ex, typename F>
inline BOOST_THREAD_FUTURE<
  typename boost::result_of<
    F(BOOST_THREAD_FUTURE<R>)>::type> BOOST_THREAD_FUTURE<R>::then(
  Ex& ex,
  BOOST_THREAD_FWD_REF(F) f) {
  typedef typename boost::result_of<
    F(BOOST_THREAD_FUTURE<R>)>::type future_type;

  BOOST_THREAD_ASSERT_PRECONDITION(
    this->future_.get() != nullptr,
    boost::future_uninitialized());

  boost::shared_ptr<
    boost::detail::shared_state_base> shared_state(this->future_);
  boost::unique_lock<boost::mutex> lock(shared_state->mutex_);

  return BOOST_THREAD_MAKE_RV_REF(
    (
      boost::detail::make_future_executor_continuation_shared_state<
        Ex,
        BOOST_THREAD_FUTURE<R>,
        future_type>(
          ex,
          lock,
          boost::move(*this),
          boost::forward<F>(f)
  )));
}
#endif // BOOST_THREAD_PROVIDES_EXECUTORS

template <typename R>
template <typename F>
inline BOOST_THREAD_FUTURE<
  typename boost::result_of<
    F(BOOST_THREAD_FUTURE<R>)>::type> BOOST_THREAD_FUTURE<R>::then(
  BOOST_THREAD_FWD_REF(F) f) {
#ifndef BOOST_THREAD_CONTINUATION_SYNC
  return this->then(this_policy(), boost::forward<F>(f));
#else // BOOST_THREAD_CONTINUATION_SYNC
  typedef typename boost::result_of<
    F(BOOST_THREAD_FUTURE<R>)>::type future_type;

  BOOST_THREAD_ASSERT_PRECONDITION(
    this->future_.get() != nullptr,
    boost::future_uninitialized());

  boost::shared_ptr<
    boost::detail::shared_state_base> shared_state(this->future_);
  boost::unique_lock<boost::mutex> lock(shared_state->mutex_);
  boost::launch policy = this->launch_policy(lock);

  if (boost::underlying_cast<int>(policy) &&
      (int)boost::launch::deferred) {
    return BOOST_THREAD_MAKE_RV_REF(
      (
        boost::detail::make_future_deferred_continuation_shared_state<
          BOOST_THREAD_FUTURE<R>,
          future_type>(
            lock,
            boost::move(*this),
            boost::forward<F>(f)
    )));
  } else {
    return BOOST_THREAD_MAKE_RV_REF(
      (
        boost::detail::make_future_async_continuation_shared_state<
          BOOST_THREAD_FUTURE<R>,
          future_type>(
            lock,
            boost::move(*this),
            boost::forward<F>(f)
    )));
  }
#endif // BOOST_THREAD_CONTINUATION_SYNC
}

template <typename R>
template <typename F>
inline BOOST_THREAD_FUTURE<
  typename boost::result_of<
    F(BOOST_THREAD_FUTURE<BOOST_THREAD_FUTURE<R> >)>::type>
  BOOST_THREAD_FUTURE<BOOST_THREAD_FUTURE<R> >::then(
  boost::launch policy,
  BOOST_THREAD_FWD_REF(F) f) {
  typedef BOOST_THREAD_FUTURE<R> R2;
  typedef typename boost::result_of<
    F(BOOST_THREAD_FUTURE<R2>)>::type future_type;

  BOOST_THREAD_ASSERT_PRECONDITION(
    this->future_.get() != nullptr,
    boost::future_uninitialized());

  boost::shared_ptr<
    boost::detail::shared_state_base> shared_state(this->future_);
  boost::unique_lock<boost::mutex> lock(shared_state->mutex_);

  if (boost::underlying_cast<int>(policy) &&
      (int)boost::launch::async) {
    return BOOST_THREAD_MAKE_RV_REF(
      (
        boost::detail::make_future_async_continuation_shared_state<
          BOOST_THREAD_FUTURE<R2>,
          future_type>(
            lock,
            boost::move(*this),
            boost::forward<F>(f)
    )));
  } else if (boost::underlying_cast<int>(policy) &&
             (int)boost::launch::sync) {
    return BOOST_THREAD_MAKE_RV_REF(
      (
        boost::detail::make_future_sync_continuation_shared_state<
          BOOST_THREAD_FUTURE<R2>,
          future_type>(
            lock,
            boost::move(*this),
            boost::forward<F>(f)
    )));
  } else if (boost::underlying_cast<int>(policy) &&
             (int)boost::launch::deferred) {
    return BOOST_THREAD_MAKE_RV_REF(
      (
        boost::detail::make_future_deferred_continuation_shared_state<
          BOOST_THREAD_FUTURE<R2>,
          future_type>(
            lock,
            boost::move(*this),
            boost::forward<F>(f)
    )));
#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
  } else if (boost::underlying_cast<int>(policy) &&
             (int)boost::launch::executor) {
    assert(this->future_->get_executor());
    typedef executor Ex;
    Ex& ex = *(this->future_->get_executor());

    return BOOST_THREAD_MAKE_RV_REF(
      (
        boost::detail::make_future_executor_continuation_shared_state<
          Ex,
          BOOST_THREAD_FUTURE<R2>,
          future_type>(
            ex,
            lock,
            boost::move(*this),
            boost::forward<F>(f)
    )));
#endif // BOOST_THREAD_PROVIDES_EXECUTORS
  } else if (boost::underlying_cast<int>(policy) &&
             (int)boost::launch::inherit) {
    boost::launch policy_ = this->launch_policy(lock);
    if (boost::underlying_cast<int>(policy_) &&
        (int)boost::launch::async) {
      return BOOST_THREAD_MAKE_RV_REF(
        (
          boost::detail::make_future_async_continuation_shared_state<
            BOOST_THREAD_FUTURE<R2>,
            future_type>(
              lock,
              boost::move(*this),
              boost::forward<F>(f)
      )));
    } else if (boost::underlying_cast<int>(policy_) &&
               (int)boost::launch::sync) {
      return BOOST_THREAD_MAKE_RV_REF(
        (
          boost::detail::make_future_sync_continuation_shared_state<
            BOOST_THREAD_FUTURE<R2>,
            future_type>(
              lock,
              boost::move(*this),
              boost::forward<F>(f)
      )));
    } else if (boost::underlying_cast<int>(policy_) &&
               (int)boost::launch::deferred) {
      return BOOST_THREAD_MAKE_RV_REF(
        (
          boost::detail::make_future_deferred_continuation_shared_state<
            BOOST_THREAD_FUTURE<R2>,
            future_type>(
              lock,
              boost::move(*this),
              boost::forward<F>(f)
      )));
#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
    } else if (boost::underlying_cast<int>(policy_) &&
               (int)boost::launch::executor) {
      assert(this->future_->get_executor());
      typedef executor Ex;
      Ex& ex = *(this->future_->get_executor());
      return BOOST_THREAD_MAKE_RV_REF(
        (
          boost::detail::make_future_executor_continuation_shared_state<
            BOOST_THREAD_FUTURE<R2>,
            future_type>(
              ex,
              lock,
              boost::move(*this),
              boost::forward<F>(f)
      )));
#endif // BOOST_THREAD_PROVIDES_EXECUTORS
    } else {
      return BOOST_THREAD_MAKE_RV_REF(
        (
          boost::detail::make_future_async_continuation_shared_state<
            BOOST_THREAD_FUTURE<R>,
            future_type>(
              lock,
              boost::move(*this),
              boost::forward<F>(f)
      )));
    }
  } else {
    return BOOST_THREAD_MAKE_RV_REF(
      (
        boost::detail::make_future_async_continuation_shared_state<
          BOOST_THREAD_FUTURE<R>,
          future_type>(
            lock,
            boost::move(*this),
            boost::forward<F>(f)
    )));
  }
}

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
template <typename R>
template <typename Ex, typename F>
inline BOOST_THREAD_FUTURE<
  typename boost::result_of<
    F(BOOST_THREAD_FUTURE<BOOST_THREAD_FUTURE<R> >)>::type>
    BOOST_THREAD_FUTURE<BOOST_THREAD_FUTURE<R> >::then(
      Ex& ex,
      BOOST_THREAD_FWD_REF(F) f) {
  typedef BOOST_THREAD_FUTURE<R> R2;
  typedef typename boost::result_of<
    F(BOOST_THREAD_FUTURE<R2>)>::type future_type;
  BOOST_THREAD_ASSERT_PRECONDITION(
    this->future_.get() != 0,
    boost::future_uninitialized());

  boost::shared_ptr<
    boost::detail::shared_state_base> shared_state(this->future_);
  boost::unique_lock<boost::mutex> lock(shared_state->mutex_);
  boost::launch policy = this->launch_policy(lock);

  if (boost::underlying_cast<int>(policy) &&
      (int)boost::launch::executor) {
    return BOOST_THREAD_MAKE_RV_REF(
      (
        boost::detail::make_future_deferred_continuation_shared_state<
          BOOST_THREAD_FUTURE<R2>,
          future_type>(
            lock,
            boost::move(*this),
            boost::forward<F>(f)
    )));
  } else {
    return BOOST_THREAD_MAKE_RV_REF(
      (
        boost::detail::make_future_sync_continuation_shared_state<
          BOOST_THREAD_FUTURE<R2>,
          future_type>(
            lock,
            boost::move(*this),
            boost::forward<F>(f)
    )));
  }
}
#endif // BOOST_THREAD_PROVIDES_EXECUTORS
template <typename R>
template <typename F>
inline BOOST_THREAD_FUTURE<
  typename boost::result_of<
    F(BOOST_THREAD_FUTURE<BOOST_THREAD_FUTURE<R> >)>::type>
    BOOST_THREAD_FUTURE<BOOST_THREAD_FUTURE<R> >::then(
      BOOST_THREAD_FWD_REF(F) f) {
#ifndef BOOST_THEAD_CONTINUATION_SYNC
  return this->then(this->launch_policy(), boost::forward<F>(f));
#else // BOOST_THREAD_CONTINUATION_SYNC
  typedef BOOST_THREAD_FUTURE<R> R2;
  typedef typename boost::result_of<
    F(BOOST_THREAD_FUTURE<R2>)>::type future_type;
  BOOST_THREAD_ASSERT_PRECONDITION(
    this->future_.get() != 0,
    boost::future_uninitialized());

  boost::shared_ptr<
    boost::detail::shared_state_base> shared_state(this->future_);
  boost::unique_lock<boost::mutes> lock(shared_state->mutex_);
  boost::launch policy = this->launch_policy(lock);

  if (boost::underlying_cast<int>(policy) &&
      (int)boost::launch::deferred) {
    return BOOST_THREAD_MAKE_RV_REF(
      (
        boost::detail::make_future_deferred_continuation_shared_state<
          BOOST_THREAD_FUTURE<R2>,
          future_type>(
            lock,
            boost::move(*this),
            boost::forward<F>(f)
    )));
  } else {
    return BOOST_THREAD_MAKE_RV_REF(
      (
        boost::detail::make_future_sync_continuation_shared_state<
          BOOST_THREAD_FUTURE<R2>,
          future_type>(
            lock,
            boost::move(*this),
            boost::forward<F>(f)
    )));
  }
#endif // BOOST_THREAD_CONTINUATION_SYNC
}

template <typename R>
template <typename F>
inline BOOST_THREAD_FUTURE<
  typename boost::result_of<
     F(boost::shared_future<R>)>::type>
  boost::shared_future<R>::then(
    boost::launch policy,
    BOOST_THREAD_FWD_REF(F) f) const {
  typedef typename boost::result_of<
    F(boost::shared_future<R>)>::type future_type;
  BOOST_THREAD_ASSERT_PRECONDITION(
    this->future_.get() != 0,
    boost::future_uninitialized());
  boost::unique_lock<boost::mutex> lock(this->future_->mutex_);

  if (boost::underlying_cast<int>(policy) &&
      (int)boost::launch::async) {
    return BOOST_MAKE_RV_REF(
      (
        boost::detail::make_shared_future_async_continuation_shared_state<
          boost::shared_future<R>,
          future_type>(
            lock,
            *this,
            boost::forward<F>(f)
    )));
  } else if (boost::underlying_cast<int>(policy) &&
             (int)boost::launch::sync) {
    return BOOST_THREAD_MAKE_RV_REF(
      (
        boost::detail::make_shared_future_sync_continuation_shared_state<
          boost::shared_future<R>,
          future_type>(
            lock,
            *this,
            boost::forward<F>(f)
    )));
  } else if (boost::underlying_cast<int>(policy) &&
             (int)boost::launch::deferred) {
    return BOOST_MAKE_RV_REF(
      (
        boost::detail::make_shared_future_deferred_continuation_shared_state<
          boost::shared_future<R>,
          future_type>(
            lock,
            *this,
            boost::forward<F>(f)
    )));
#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
  } else if (boost::underlying_cast<int>(policy) &&
             (int)boost::launch::executor) {
    typedef executor Ex;
    Ex& ex = *(this->future_->get_executor());
    return BOOST_THREAD_MAKE_RV_REF(
      (
        boost::detail::make_shared_future_executor_continuation_shared_state<
          boost::shared_future<R>,
          future_type>(
            ex,
            lock,
            *this,
            boost::forward<F>(f)
    )));
#endif // BOOST_THREAD_PROVIDES_EXECUTORS
  } else if (boost::underlying_cast<int>(policy) &&
             (int)boost::launch::inherit) {
    boost::launch policy_ = this->launch_policy(lock);
    if (boost::underlying_cast<int>(policy_) &&
        (int)boost::launch::async) {
      return BOOST_THREAD_MAKE_RV_REF(
        (
          boost::detail::make_shared_future_async_continuation_shared_state<
            boost::shared_future<R>,
            future_type>(
              lock,
              *this,
              boost::forward<F>(f)
      )));
    } else if (boost::underlying_cast<int>(policy_) &&
               (int)boost::launch::sync) {
      return BOOST_THREAD_MAKE_RV_REF(
        (
          boost::detail::make_shared_future_sync_continuation_shared_state<
            boost::shared_future<R>,
            future_type>(
              lock,
              *this,
              boost::forward<F>(f)
          )));
    } else if (boost::underlying_cast<int>(policy_) &&
               (int)boost::launch::deferred) {
      return BOOST_THREAD_MAKE_RV_REF(
        (
          boost::detail::make_shared_future_deferred_continuation_shared_state<
            boost::shared_future<R>,
            future_type>(
              lock,
              *this,
              boost::forward<F>(f)
      )));
#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
    } else if (boost::underlying_cast<int>(policy_) &&
               (int)boost::launch::executor) {
      typedef executor Ex;
      Ex& ex = *(this->future_->get_executor());
      return BOOST_THREAD_MAKE_RV_REF(
        (
          boost::detail::make_shared_future_executor_continuation_shared_state<
            boost::shared_future<R>,
            future_type>(
              ex,
              lock,
              *this,
              boost::forward<F>(f)
      )));
#endif // BOOST_THREAD_PROVIDES_EXECUTORS
    } else {
      return BOOST_THREAD_MAKE_RV_REF(
        (
          boost::detail::make_shared_future_async_continuation_shared_state<
            boost::shared_state<R>,
            future_type>(
              lock,
              *this,
              boost::forward<F>(f)
      )));
    }
  } else {
    return BOOST_THREAD_MAKE_RV_REF(
      (
        boost::detail::make_shared_future_async_continuation_shared_state<
          boost::shared_future<R>,
          future_type>(
            lock,
            *this,
            boost::forward<F>(f)
    )));
  }
}


} // boost

#endif // BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
#endif // CONTINUATION_IPP
