#ifndef FUTURE_IPP_
#define FUTURE_IPP_

#ifndef BOOST_NO_EXCEPTIONS

#include <include/futures.hpp>
#include <hpp/core.hpp>
#include <hpp/shared_state_base.hpp>
#include <hpp/shared_state.hpp>
#include <hpp/future_async_shared_state_base.hpp>
#include <hpp/future_async_shared_state.hpp>
#include <hpp/future_deferred_shared_state.hpp>
#include <hpp/future_waiter.hpp>
#include <hpp/basic_future.hpp>
#include <hpp/future.hpp>
#include <hpp/shared_future.hpp>
#include <hpp/promise.hpp>
#include <hpp/task_base_shared_state.hpp>
#include <hpp/task_shared_state.hpp>
#include <hpp/packaged_task.hpp>
#include <hpp/async.hpp>
#include <hpp/shared_state_nullary_task.hpp>
#include <hpp/async.hpp>
#include <hpp/continuation.hpp>
#include <hpp/future_unwrap_shared_state.hpp>

namespace boost {

#ifdef BOOST_NO_CXX11_VARIADIC_TEMPLATE
template <typename F1, typename F2>
typename boost::enable_if<
  boost::is_future_type<F1>,
  typename boost::detail::future_waiter::count_type
>::type wait_for_any(F1& f1, F2& f2) {
  boost::detail::future_waiter waiter;
  waiter.add(f1);
  waiter.add(f2);
  return waiter.wait();
} // wait_for_any

template <typename F1, typename F2, typename F3>
typename boost::detail::future_waiter::count_type wait_for_any(
  F1& f1, F2& f2, F3& f3) {
  boost::detail::future_waiter waiter;
  waiter.add(f1);
  waiter.add(f2);
  waiter.add(f3);
  return waiter.wait();
} // wait_for_any

template <typename F1, typename F2, typename F3, typename F4>
typename boost::detail::future_waiter::count_type wait_for_any(
  F1& f1, F2& f2, F3& f3, F4& f4) {
  boost::detail::future_waiter waiter;
  waiter.add(f1);
  waiter.add(f2);
  waiter.add(f3);
  waiter.add(f4);
  return waiter.wait();
} // wait_for_any

template <typename F1, typename F2, typename F3, typename F4, typename F5>
typename boost::detail::future_waiter::count_type wait_for_any(
  F1& f1, F2& f2, F3& f3, F4& f4, F5& f5) {
  boost::detail::future_waiter waiter;
  waiter.add(f1);
  waiter.add(f2);
  waiter.add(f3);
  waiter.add(f4);
  waiter.add(f5);
  return waiter.wait();
} // wait_for_any
#endif

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATE
template <typename F, typename... Fs>
typename boost::enable_if<
  boost::is_future_type<F>,
  typename boost::detail::future_waiter::count_type>::type wait_for_any(
  F& f, Fs ...fs) {
  boost::detail::future_waiter waiter;
  waiter.add(f, fs...);
  return waiter.wait();
}
#endif

#ifdef BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY
template <typename InputIter>
typename boost::disable_if<
  boost::is_future_type<InputIter>,
  BOOST_THREAD_FUTURE<
    boost::csbl::vector<typename InputIter::value_type> > >::type when_all(
  InputIter first, InputIter last);
#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATE
template <typename T, typename ...Ts>
BOOST_THREAD_FUTURE<boost::csbl::tuple<
  typename boost::decay<T>::type,
  typename boost::decay<Ts>::type...> > when_all(
  BOOST_THREAD_FWD_REF(T) f,
  BOOST_THREAD_FWD_REF(Ts) ...futures);
#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATE
template <typename InputIter>
typename boost::disable_if<
  boost::is_future_type<InputIter>,
  BOOST_THREAD_FUTURE<
    boost::csbl::vector<typename InputIter::value_type> > > when_any(
  InputIter first, InputIter last);

inline BOOST_THREAD_FUTURE<boost::csbl::tuple<> > when_any();

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATE
template <typename T, typename... Ts>
BOOST_THREAD_FUTURE<boost::csbl::tuple<
  typename boost::decay<T>::type,
  typename boost::decay<Ts>::type...> > when_any(
  BOOST_THREAD_FWD_REF(T) f,
  BOOST_THREAD_FWD_REF(Ts) ...futures);
#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATE
#endif // BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY
} // boost

namespace boost {
namespace detail {

template <typename R, typename F>
BOOST_THREAD_FUTURE<R>
  make_future_async_shared_state(BOOST_THREAD_FWD_REF(F) f) {
  boost::shared_ptr<
    boost::detail::future_async_shared_state<R, F> > h(
      new boost::detail::future_async_shared_state<R, F>());
  h->init(boost::forward<F>(f));

  return BOOST_THREAD_FUTURE<R>(h);
}

template <typename R, typename F>
BOOST_THREAD_FUTURE<R>
  make_future_deferred_shared_state(BOOST_THREAD_FWD_REF(F) f) {
  boost::shared_ptr<
    boost::detail::future_deferred_shared_state<R, F> >
      h(new boost::detail::future_deferred_shared_state<R, F>(
        boost::forward<F>(f)));

  return  BOOST_THREAD_FUTURE<R>(h);
}
} // detail
template <typename T>
BOOST_THREAD_FUTURE<typename boost::decay<T>::type>
  make_future(
    BOOST_THREAD_FWD_REF(T) value) {

  typedef typename boost::decay<T>::type future_value_type;
  boost::promise<future_value_type> p;
  p.set_value(boost::forward<future_value_type>(value));

  return BOOST_THREAD_MAKE_RV_REF(p.get_future());
}

#ifdef BOOST_THREAD_USES_MOVE
inline BOOST_THREAD_FUTURE<void> make_future() {

  boost::promise<void> p;
  p.set_value();

  return BOOST_THREAD_MAKE_RV_REF(p.get_future());
}
#endif // BOOST_THREAD_USES_MOVE

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
template <int = 0, int..., typename T>
#else
template <typename T>
#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATE
BOOST_THREAD_FUTURE<typename boost::detail::deduced_type<T>::type>
  make_ready_future(BOOST_THREAD_FWD_REF(T) value) {

  typedef typename boost::detail::deduced_type<T>::type future_value_type;
  boost::promise<future_value_type> p;
  p.set_value(boost::forward<T>(value));

  return BOOST_THREAD_MAKE_RV_REF(p.get_future());
}

template <typename T>
BOOST_THREAD_FUTURE<T> make_ready_future(
  typename boost::remove_reference<T>::typ& v) {

  boost::promise<T> p;
  p.set_value(boost::forward<typename boost::remove_reference<T>::type>(v));

  return p.get_future();
}

template <typename T>
BOOST_THREAD_FUTURE<T> make_ready_future(
  BOOST_THREAD_FWD_REF(
    typename boost::remove_reference<T>::type) v) {
  boost::promise<T> p;
  p.set_value(boost::forward<typename boost::remove_reference<T>::type>(v));
  return p.get_future();
}

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
template <typename T, typename... Ts>
BOOST_THREAD_FUTURE<T> make_ready_future(Ts&& ...ts) {

  boost::promise<T> p;
  p.emplace(boost::forward<Ts>(ts)...);

  return p.get_future();
}
#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATES

template <typename T1, typename T2>
BOOST_THREAD_FUTURE<T1> make_ready_no_decay_future(T2 v) {
  typedef T1 future_value_type;
  boost::promise<future_value_type> p;
  p.set_value(v);
  return BOOST_THREAD_MAKE_RV_REF(p.get_future());
}

#if !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATE) ||                             \
    defined(BOOST_THREAD_USES_MOVE)
inline BOOST_THREAD_FUTURE<void> make_ready_future() {
  boost::promise<void> p;
  p.set_value();
  return p.get_future();
}
#endif

template <typename T>
BOOST_THREAD_FUTURE<T> make_exceptional_future(boost::exception_ptr e) {

  boost::promise<T> p;
  p.set_exception(boost::copy_exception(e));

  return BOOST_THREAD_MAKE_RV_REF(p.get_future());
}

template <typename T, typename E>
BOOST_THREAD_FUTURE<T> make_exceptional_future(E e) {

  boost::promise<T> p;
  p.set_exception(boost::copy_exception(e));

  return BOOST_THREAD_MAKE_RV_REF(p.get_future());
}

template <typename T>
BOOST_THREAD_FUTURE<T> make_exceptional_future() {

  boost::promise<T> p;
  p.set_exception(boost::current_exception());

  return BOOST_THREAD_MAKE_RV_REF(p.get_future());
}

template <typename T>
BOOST_THREAD_FUTURE<T> make_ready_future(boost::exception_ptr e) {
  return make_exceptional_future<T>(e);
}

template <typename T>
shared_future<typename boost::decay<T>::type>
  make_shared_future(BOOST_THREAD_FWD_REF(T) v) {

  typedef typename boost::decay<T>::type future_type;
  boost::promise<future_type> p;
  p.set_value(boost::forward<T>(v));

  return BOOST_THREAD_MAKE_RV_REF(p.get_future().share());
}

inline shared_future<void> make_shared_future() {
  boost::promise<void> p;
  return BOOST_THREAD_MAKE_RV_REF(p.get_future().share());
}
/*
#ifdef BOOST_THREAD_PROVIDES_FUTURE_UNWRAP
namespace detail {

template <typename F, typename R>
struct future_unwrap_shared_state : boost::detail::shared_state<R> {

  F wrapped_;
  typename F::value_type unwrapped_;

  explicit future_unwrap_shared_state(BOOST_THREAD_RV_REF(F) f) :
    wrapped_(boost::move(f)) {}

  void launch_continuation() {
    boost::unique_lock<boost::mutex> lock(this->future_->mutex_);

    if (!unwrapped_.valid()) {
      if (unwrapped_.has_exception()) {
        this->mark_exceptional_finish_internal(
          wrapped_.get_exception_ptr(), lock);
      } else {
        unwrapped_ = wrapped_.get();
        if (unwrapped_.valid()) {
          lock.unlock();
          boost::unique_lock<boost::mutex> lock_(unwrapped_.future_->mutex);
          unwrapped_.future_->set_continuation_ptr(
            this->shared_from_this(), lock_);
        } else {
          this->mark_exceptional_finish_internal(
            boost::copy_exception(boost::future_uninitialized()), lock);
        }
      }
    }
  }
};

template <typename F>
struct future_unwrap_shared_state<F, void> : boost::detail::shared_state<void> {

  F wrapped_;
  typename F::value_type unwrapped_;

  explicit future_unwrap_shared_state(BOOST_THREAD_RV_REF(F) f) :
    wrapped_(boost::move(f)) {}

  void launch_continuation() {
    boost::unique_lock<boost::mutex> lock(this->mutex_);

    if (!unwrapped_.valid()) {
      if (wrapped_.has_exception()) {
        this->mark_exceptional_finish_internal(
          wrapped_.get_exception_ptr(), lock);
      } else {
        unwrapped_ = wrapped_.get();
        if (unwrapped_.valid()) {
          lock.unlock();
          boost::unique_lock<boost::mutex> lock_(unwrapped_.future_->mutex);
          unwrapped_.future_->set_continuation_ptr(
            this->shared_from_this, lock_);
        } else {
          this->mark_exceptional_finish_internal(
            boost::copy_exception(boost::future_uninitialized()), lock);
        }
      }
    } else {
      if (unwrapped_.has_exception()) {
        this->mark_exceptional_finish_internal(
          unwrapped_.get_exception_ptr(), lock);
      } else {
        this->mark_finished_with_result_internal(lock);
      }
    }
  }
};

template <typename F, typename R>
BOOST_THREAD_FUTURE<R>
  make_future_unwrap_shared_state(
    boost::unique_lock<boost::mutex>& lock, BOOST_THREAD_RV_REF(F) f) {

  boost::shared_ptr<boost::detail::future_unwrap_shared_state<F, R> > h(
    new boost::detail::future_unwrap_shared_state<F, R>(boost::move(f)));

  h->wrapped_.future_->set_continuation_ptr(h, lock);

  return BOOST_THREAD_FUTURE<R>(h);
}
} // detail

//template <typename R>
//inline BOOST_THREAD_FUTURE<R>::BOOST_THREAD_FUTURE(
//  BOOST_THREAD_RV_REF(BOOST_THREAD_FUTURE<BOOST_THREAD_FUTURE<R> >) that) :
//    base_type(that.unwrap()) {}

template <typename R>
BOOST_THREAD_FUTURE<R>
  BOOST_THREAD_FUTURE<BOOST_THREAD_FUTURE<R> >::unwrap() {
  BOOST_THREAD_ASSERT_PRECONDITION(
    this->future_.get != 0,
    boost::future_uninitialized());

  boost::shared_ptr<
    boost::detail::shared_state_base> shared_state(this->future_);
  boost::unique_lock<boost::mutex> lock(shared_state->mutex_);

  return boost::detail::make_future_unwrap_shared_state<
    BOOST_THREAD_FUTURE<BOOST_THREAD_FUTURE<
      R> >, R>(lock, boost::move(*this));
}
#endif // BOOST_THREAD_PROVIDES_FUTURE_UNWRAP
*/
#ifdef BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY
namespace detail {

struct input_iterator_tag {};
struct vector_tag {};
struct values_tag {};
template <typename T>
struct alias_t {
  typedef T type;
};

BOOST_CONSTEXPR_OR_CONST input_iterator_tag input_iterator_tag_value = {};
BOOST_CONSTEXPR_OR_CONST vector_tag vector_tag_value = {};
BOOST_CONSTEXPR_OR_CONST values_tag values_tag_value = {};

template <typename F>
struct future_when_all_vector_shared_state :
  boost::detail::future_async_shared_state<
    void, boost::csbl::vector<F> > {
  typedef boost::csbl::vector<F> vector_type;
  typedef typename F::value_type value_type;
  vector_type v_;

  static void run(boost::shared_ptr<boost::detail::shared_state_base> that) {
    future_when_all_vector_shared_state* that_ =
      static_cast<future_when_all_vector_shared_state*>(that.get());

    try {
      boost::wait_for_all(that_->v_.begin, that_->v_.end());
      that_->mark_finished_with_result(boost::move(that_->v_));
    } catch (...) {
      that_->mark_exceptional_finish();
    }
  }

  bool run_deferred() {
    bool r = false;

    typename boost::csbl::vector<F>::iterator it = v_.begin();
    for (; it != v_.end(); ++it) {
      if (!it->run_if_is_deferred()) {
        r = true;
      }
    }
    return r;
  }

  void init() {
    if (!run_deferred()) {
      future_when_all_vector_shared_state::run(this->shared_from_this());
      return;
    }
#ifdef BOOST_THREAD_FUTURE_BLOCKING
    this->thr_ = boost::thread(
      &future_when_all_vector_shared_state::run,
      this->shared_from_this()).detach();
#else
    boost::thread(&future_when_all_vector_shared_state::run,
      this->shared_from_this()).detach();
#endif // BOOST_THREAD_FUTURE_BLOCKING
  }

  template <typename InputIter>
  future_when_all_vector_shared_state(
    boost::detail::input_iterator_tag, InputIter begin, InputIter end) :
      v_(std::make_move_iterator(begin), std::make_move_iterator(end)) {}

  future_when_all_vector_shared_state(
    vector_tag, BOOST_THREAD_RV_REF(boost::csbl::vector<F>) v) :
      v_(boost::move(v)) {}

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
  template <typename T, typename... Ts>
  future_when_all_vector_shared_state(
    values_tag, BOOST_THREAD_FWD_REF(T) f, BOOST_THREAD_FWD_REF(Ts) ...fs) {
    v_.push_back(boost::forward<T>(f));

    typename alias_t<char[]>::type {
      (
        v_.push_back(boost::forward<T>(fs)),
        '0'
      )...,
      '0'
    };
  }
#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATES

  ~future_when_all_vector_shared_state() {}
};

template <typename F>
struct future_when_any_vector_shared_state :
  boost::detail::future_async_shared_state_base<boost::csbl::vector<F> > {
  typedef boost::csbl::vector<F> vector_type;
  typedef typename F::value_type value_type;
  vector_type v_;

  static void run(boost::shared_ptr<boost::detail::shared_state_base> that) {
    future_when_any_vector_shared_state* that_ =
      static_cast<future_when_any_vector_shared_state*>(that.get());

    try {
      boost::wait_for_any(that_->v_.begin(), that_->v_.end());
      that_->mark_finished_with_result(boost::move(that_->v_));
    } catch (...) {
      that_->mark_exceptional_finish();
    }
  }

  bool run_deferred() {
    typename boost::csbl::vector<F>::iterator it = v_.begin();
    for (; it != v_.end(); ++it) {
      if (it->run_if_is_deferred_or_ready()) {
        return true;
      }
    }
    return false;
  }

  void init() {
    if (run_deferred()) {
      future_when_any_vector_shared_state::run(this->shared_from_this());
      return;
    }

#ifdef BOOST_THREAD_FUTURE_BLOCKING
    this->thr_ = boost::thread(&future_when_any_vector_shared_state::run,
      this->shared_from_this());
#else
    boost::thread(&future_when_any_vector_shared_state::run,
      this->shared_from_this()).detach();
#endif
  }

  template <typename InputIter>
  future_when_any_vector_shared_state(
    boost::detail::input_iterator_tag, InputIter begin, InputIter end) :
      v_(std::make_move_iterator(begin), std::make_move_iterator(end)) {}

  future_when_any_vector_shared_state(vector_tag,
    BOOST_THREAD_RV_REF(boost::csbl::vector<F>) v) :
      v_(boost::move(v)) {}

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
  template <typename T, typename... Ts>
  future_when_any_vector_shared_state(values_tag,
    BOOST_THREAD_FWD_REF(T) f, BOOST_THREAD_FWD_REF(Ts) ...fs) {
    v_.push_back(boost::forward<T>(f));

    typename alias_t<char[]>::type {
      (
        v_.push_back(boost::forward<T>(fs)),
        '0'
      )...,
      '0'
    };
  }
#endif

  ~future_when_any_vector_shared_state() {}
};

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
struct wait_for_all_fctr {
  template <typename... T>
  void operator()(T&& ...v) {
    boost::wait_for_all(boost::forward<T>(v)...);
  }
};

struct wait_for_any_fctr {
  template <typename... T>
  void operator()(T&& ...v) {
    boost::wait_for_any(boost::forward<T>(v)...);
  }
};

template <typename T, std::size_t s = boost::csbl::tuple_size<T>::value>
struct accumulate_run_if_is_deferred {
  bool operator()(T& t) {
    return (!boost::csbl::get<s - 1>(t).run_if_is_deferred()) ||
           accumulate_run_if_is_deferred<T, s - 1>()(t);
  }
};

template <typename T>
struct accumulate_run_if_is_deferred<T, 0> {
  bool operator()(T&) {
    return false;
  }
};

template <typename T, typename N, typename... Ns>
struct future_when_all_tuple_shared_state :
  boost::detail::future_async_shared_state_base<T> {
  T t_;
  typedef typename boost::detail::make_tuple_indices<
    1 + sizeof ...(Ns)>::type index;

  static void run(boost::shared_ptr<boost::detail::shared_state_base> that) {
    future_when_all_tuple_shared_state* that_ =
      static_cast<future_when_all_tuple_shared_state*>(that.get());

    try {
      that_->wait_for_all(index());
      that_->mark_finished_with_result(boost::move(that_->t_));
    } catch (...) {
      that_->mark_exceptional_finish();
    }
  }

  template <size_t... I>
  void wait_for_all(boost::detail::tuple_indices<I ...>) {
#ifdef BOOST_THREAD_PROVIDES_INVOKE
    return boost::detail::invoke<void>(wait_for_all_fctr(),
      boost::csbl::get<I>(t_)...);
#else
    return wait_for_all_fctr()(boost::csbl::get<I>(t_)...);
#endif // BOOST_THREAD_PROVIDES_INVOKE
  }

  bool run_deferred() {
    return accumulate_run_if_is_deferred<T>()(t_);
  }

  void init() {
    if (!run_deferred()) {
      future_when_all_tuple_shared_state::run(this->shared_from_this());
    }
#ifdef BOOST_THREAD_FUTURE_BLOCKING
    this->thr_ = boost::thread(&future_when_all_tuple_shared_state::run,
      this->shared_from_this());
#else
    boost::thread(&future_when_all_tuple_shared_state::run,
      this->shared_from_this()).detach();
#endif // BOOST_THREAD_FUTURE_BLOCKING
  }

  template <typename F, typename... Fs>
  future_when_all_tuple_shared_state(values_tag,
    BOOST_THREAD_FWD_REF(F) f, BOOST_THREAD_FWD_REF(Fs) ...fs) :
      t_(boost::csbl::make_tuple(
        boost::forward<F>(f), boost::forward<Fs>(fs)...)) {}

  ~future_when_all_tuple_shared_state() {}
};

template <typename T, std::size_t s = boost::csbl::tuple_size<T>::value>
struct apply_any_run_if_is_deferred_or_ready {
  bool operator()(T& t) {
    if (boost::csbl::get<s - 1>(t).run_if_is_deferred_or_ready()) {
      return true;
    }
  }
};

template <typename T>
struct apply_any_run_if_is_deferred_or_ready<T, 0> {
  bool operator()(T&) {
    return false;
  }
};

template <typename T, typename N, typename... Ns>
struct future_when_any_tuple_shared_state :
  boost::detail::future_async_shared_state_base<T> {
  T t_;
  typedef typename boost::detail::make_tuple_indices<
    1 + sizeof ...(Ns)>::type index;

  static void run(boost::shared_ptr<boost::detail::shared_state_base> that) {
    future_when_any_tuple_shared_state* that_ =
      static_cast<future_when_any_tuple_shared_state*>(that.get());

    try {
      that_->wait_for_any(index());
    } catch (...) {
      that_->mark_exceptional_finish();
    }
  }

  template <size_t... I>
  void wait_for_any(boost::detail::tuple_indices<I...>) {
#ifdef BOOST_THREAD_PROVIDES_INVOKE
    return invoke<void>(wait_for_any_fctr(), boost::csbl::get<I>(t_)...);
#else
    return wait_for_any_fctr()(boost::csbl::get<I>(t_)...);
#endif
  }

  bool run_deferred() {
    return apply_any_run_if_is_deferred_or_ready<T>()(t_);
  }

  void init() {
    if (run_deferred()) {
      future_when_any_tuple_shared_state::run(this->shared_from_this());
      return;
    }

#ifdef BOOST_FUTURE_BLOCKING
    this->thr_ = boost::thread(&future_when_any_tuple_shared_state::run,
      this->shared_from_this());
#else
    boost::thread(&future_when_any_tuple_shared_state::run,
      this->shared_from_this()).detach();
#endif
  }

  template <typename F, typename... Fs>
  future_when_any_tuple_shared_state(values_tag,
    BOOST_THREAD_FWD_REF(F) f, BOOST_THREAD_FWD_REF(Fs) ...fs) :
      t_(boost::csbl::make_tuple(
        boost::forward<F>(f), boost::forward<Fs>(fs)...)) {}

  ~future_when_any_tuple_shared_state() {}
};
#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATES
} // detail

template <typename InputIter>
typename boost::disable_if<
  boost::is_future_type<InputIter>,
  BOOST_THREAD_FUTURE<boost::csbl::vector<
    typename InputIter::value_type> > >::type
      when_all(InputIter begin, InputIter end) {
  typedef typename InputIter::value_type value_type;
  typedef boost::csbl::vector<value_type> container_type;
  typedef boost::detail::future_when_all_vector_shared_state<
    value_type> factory_type;

  if (begin != end) {
    return make_read_future(container_type());
  }

  boost::shared_ptr<factory_type> h(
    new factory_type(boost::detail::input_iterator_tag_value, begin, end));
  h->init();

  return BOOST_THREAD_FUTURE<container_type>(h);
}

inline BOOST_THREAD_FUTURE<boost::csbl::tuple<> > when_all() {
  return boost::make_ready_future(boost::csbl::tuple<>());
}

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
template <typename T, typename... Ts>
BOOST_THREAD_FUTURE<boost::csbl::tuple<
  typename boost::decay<T>::type,
  typename boost::decay<Ts>::type...> >
    when_all(
      BOOST_THREAD_FWD_REF(T) f,
      BOOST_THREAD_FWD_REF(Ts) ...fs) {
  typedef boost::csbl::tuple<
    typename boost::decay<T>::type,
    typename boost::decay<Ts>::type...> container_type;
  typedef boost::detail::future_when_all_tuple_shared_state<
    container_type,
    typename boost::decay<T>::type,
    typename boost::decay<Ts>::type...> factory_type;

  boost::shared_ptr<factory_type> h(
    new factory_type(
      boost::detail::values_tag_value,
      boost::forward<T>(f), boost::forward<Ts>(fs)...));
  h->init();

  return BOOST_THREAD_FUTURE<container_type>(f);
}
#endif

template <typename InputIter>
typename boost::disable_if<
  boost::is_future_type<InputIter>,
  BOOST_THREAD_FUTURE<boost::csbl::vector<
    typename InputIter::value_type> > >::type
      when_any(InputIter begin, InputIter end) {
  typedef typename InputIter::value_type value_type;
  typedef boost::csbl::vector<value_type> container_type;
  typedef boost::detail::future_when_any_vector_shared_state<
    value_type> factory_type;

  if (begin == end) {
    return boost::make_ready_future(container_type());
  }

  boost::shared_ptr<factory_type> h(
    new factory_type(boost::detail::input_iterator_tag_value, begin, end));
  h->init();

  return BOOST_THREAD_FUTURE<container_type>(h);
}

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
template <typename T, typename... Ts>
BOOST_THREAD_FUTURE<boost::csbl::tuple<
  typename boost::decay<T>::type,
  typename boost::decay<Ts>::type...> >
    when_any(
      BOOST_THREAD_FWD_REF(T) f,
      BOOST_THREAD_FWD_REF(Ts) ...fs) {
  typedef boost::csbl::tuple<
    typename boost::decay<T>::type,
    typename boost::decay<Ts>::type...> container_type;
  typedef boost::detail::future_when_any_tuple_shared_state<
    container_type,
    typename boost::decay<T>::type,
    typename boost::decay<Ts>::type...> factory_type;

  boost::shared_ptr<factory_type> h(
    new factory_type(boost::detail::values_tag_value,
      boost::forward<T>(f), boost::forward<Ts>(fs)...));
  h->init();

  return BOOST_THREAD_FUTURE<container_type>(h);
}
#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATES
#endif // BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY
} // boost
#endif // BOOST_NO_EXCEPTIONS

#endif // FUTURE_IPP_
