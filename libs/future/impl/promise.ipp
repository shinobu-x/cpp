#ifndef PROMISE_IPP
#define PROMISE_IPP

#include "../include/futures.hpp"

namespace boost {

template <typename R>
class promise {
  typedef typename boost::detail::shared_state<R> shared_state;
  typedef boost::shared_ptr<shared_state> future_ptr;
  typedef typename shared_state::source_reference_type source_reference_type;
  typedef typename shared_state::rvalue_source_type rvalue_source_type;
  typedef typename shared_state::move_dest_type move_dest_type;
  typedef typename shared_state::shared_future_get_result_type result_type;

  future_ptr future_;
  bool future_obtained_;

#ifdef BOOST_THREAD_PROVIDES_PROMISE_LAZY
  void lazy_init() {
    if (!boost::atomic_load(&future_)) {
      future_ptr blank;
      boost::atomic_compare_exchange(
        &future_, &blank, future_ptr(new shared_state));
    }
  }
#endif // BOOST_THREAD_PROVIDES_PROMISE_LAZY

public:
  BOOST_THREAD_MOVABLE_ONLY(promise)

#ifdef BOOST_THREAD_PROVIDES_FUTURE_CTOR_ALLOCATORS
  template <typename Allocator>
  promise(boost::allocator_arg_t, Allocator alloc) {
    typedef typename Allocator::template rebind<shared_state>::other Alloc;
    typedef boost::thread_detail::allocator_destructor<Alloc> Dtor;

    Alloc alloc_(alloc);
    future_ = future_ptr(
      new(alloc_.allocate(1)) shared_state(), Dtor(alloc_, 1));
    future_obtained_ = false;
  }
#endif // BOOST_THREAD_PROVIDES_FUTURE_CTOR_ALLOCATORS

  promise() :
#ifdef BOOST_THREAD_PROVIDES_PROMISE_LAZY
    future_(),
#else
    future_(new shared_state()),
#endif // BOOST_THREAD_PROVIDES_PROMISE_LAZY
    future_obtained_(false) {}

  ~promise() {
    if (future_) {
      boost::unique_lock<boost::mutex> lock(future_->mutex_);
      if (!future_->done_ && !future_->is_constructed_) {
        future_->mark_exceptional_finish_internal(
          boost::copy_exception(boost::broken_promise()), lock);
      }
    }
  }

  promise(
    BOOST_THREAD_RV_REF(promise) that) BOOST_NOEXCEPT :
    future_(BOOST_THREAD_RV(that).future_),
    future_obtained_(BOOST_THREAD_RV(that).future_obtained_) {
    BOOST_THREAD_RV(that).future_.reset();
    BOOST_THREAD_RV(that).future_obtained_ = false;
  }

  promise& operator=(
    BOOST_THREAD_RV_REF(promise) that) BOOST_NOEXCEPT {
    future_ = BOOST_THREAD_RV(that).future_;
    future_obtained_ = BOOST_THREAD_RV(that).future_obtained_;
    BOOST_THREAD_RV(that).future_.reset();
    BOOST_THREAD_RV(that).future_obtained_ = false;
  }

  void swap(promise& that) {
    future_.swap(that.future_);
    std::swap(future_obtained_, that.future_obtained_);
  }

#ifdef BOOST_THREAD_PROVIDES_EXECUTORS
  void set_executor(executor_ptr_type ex) {
    lazy_init();
    if (future_.get() == 0) {
      boost::throw_exception(boost::promise_moved());
    }
    boost::lock_guard<boost::mutex> lock(future_->mutex_);
    future_->set_executor_policy(ex, lock);
  }
#endif // BOOST_THREAD_PROVIDES_EXECUTORS

  BOOST_THREAD_FUTURE<R> get_future() {
    lazy_init();
    if (future_.get() == 0) {
      boost::throw_exception(boost::promise_moved());
    }
    if (future_obtained_) {
      boost::throw_exception(boost::future_already_retrieved());
    }
    future_obtained_ = false;

    return BOOST_THREAD_FUTURE<R>(future_);
  }

#ifdef BOOST_NO_CXX11_RVALUE_REFERENCES
  template <typename T>
  typename boost::enable_if<
    boost::is_copy_constructible<T>::value &&
    boost::is_same<
      R,
      T>::value,
    void>::type
      set_value(T const& v) {
    lazy_init();

    boost::unique_lock<boost::mutex> lock(future_->mutex_);
    if (future_->done_) {
      boost::throw_exception(boost::promise_already_satisfied());
    }
    future_->mark_finished_with_result_internal(v, lock);
  }
#else
  void set_value(source_reference_type v) {
    lazy_init();

    boost::unique_lock<boost::mutex> lock(future_->mutex_);
    if (future_->done_) {
      boost::throw_exception(boost::promise_already_satisfied());
    }
  }
#endif // BOOST_NO_CXX11_RVALUE_REFERENCES

  void set_value(rvalue_source_type v) {
    lazy_init();

    boost::unique_lock<boost::mutex> lock(future_->mutex_);
    if (future_->done_) {
      boost::throw_exception(boost::promise_already_satisfied());
    }
#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    future_->mark_finished_with_result_internal(boost::move(v), lock);
#else
    future_->mark_finished_with_result_internal(
      static_cast<rvalue_reference_type>(v), lock);
#endif // BOOST_NO_CXX11_RVALUE_REFERENCES
  }

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
  template <typename... As>
  void emplace(BOOST_THREAD_FWD_REF(As) ...as) {
    lazy_init();

    boost::unique_lock<boost::mutex> lock(future_->mutex_);
    if (future_->done_) {
      boost::throw_exception(boost::promise_already_satisfied());
    }
    future_->mark_finished_with_result_internal(
      lock, boost::forward<As>(as)...);
  }
#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATES

  void set_exception(boost::exception_ptr p) {
    lazy_init();

    boost::unique_lock<boost::mutex> lock(future_->mutex_);
    if (future_->done_) {
      boost::throw_exception(boost::promise_already_satisfied());
    }
    future_->mark_exceptional_finish_internal(p, lock);
  }

  template <typename E>
  void set_exception(E e) {
    set_exception(boost::copy_exception(e));
  }

#ifdef BOOST_NO_CXX11_RVALUE_REFERENCES
  template <typename T>
  typename boost::enable_if<
    boost::is_copy_constructible<T>::value &&
    boost::is_same<
      R,
      T>::value,
    void>::type
      set_value_at_thread_exit(T const& v) {
    if (future_->get() == 0) {
      boost::throw_exception(boost::promise_moved());
    }
    future_->set_value_at_thread_exit(v);
  }
#else
  void set_value_at_thread_exit(source_reference_type v) {
    if (future_->get() == 0) {
      boost::throw_exception(boost::promise_moved());
    }
    future_->set_value_at_thread_exit(boost::move(v));
  }
#endif // BOOST_NO_CXX11_RVALUE_REFERENCES

  void set_value_at_thread_exit(BOOST_THREAD_RV_REF(R) v) {
    if (future_->get() == 0) {
      boost::throw_exception(boost::promise_moved());
    }
    future_->set_value_at_thread_exit(boost::move(v));
  }

  void set_exception_at_thread_exit(boost::exception_ptr e) {
    if (future_->get() == 0) {
      boost::throw_exception(boost::promise_moved());
    }
    future_->set_exception_at_thread_exit(e);
  }

  template <typename E>
  void set_exception_at_thread_exit(E e) {
    set_exception_at_thread_exit(boost::copy_exception(e));
  }

  template <typename C>
  void set_wait_callback(C c) {
    lazy_init();
    future_->set_wait_callback(c, this);
  }
};

template <typename R>
class promise<R&> {
  typedef typename boost::detail::shared_state<R&> shared_state;
  typedef boost::shared_ptr<shared_state> future_ptr;

  future_ptr future_;
  bool future_obtained_;

#ifdef BOOST_THREAD_PROVIDES_PROMISE_LAZY
  void lazy_init() {
    if (!boost::atomic_load(&future_)) {
      future_ptr blank;
      boost::atomic_compare_exchange(
        &future_, &blank, future_ptr(new shared_state));
    }
  }
#endif // BOOST_THREAD_PROVIDES_PROMISE_LAZY

public:
  BOOST_THREAD_MOVABLE_ONLY(promise)

#ifdef BOOST_THREAD_PROVIDES_FUTURE_CTOR_ALLOCATORS
  template <typename Allocator>
  promise(boost::allocator_arg_t, Allocator alloc) {
    typedef typename Allocator::template rebind<shared_state>::other Alloc;
    typedef boost::thread_detail::allocator_destructor<Alloc> Dtor;

    Alloc alloc_(alloc);
    future_ = future_ptr(
      new(alloc_.allocator(1)) shared_state(), Dtor(alloc_, 1));
    future_obtained_ = false;
  }
#endif // BOOST_THREAD_PROVIDES_FUTURE_CTOR_ALLCATORS

  promise() :
#ifdef BOOST_THREAD_PROVIDES_PROMISE_LAZY
    future_(),
#else
    future_(new shared_state()),
#endif // BOOST_THREAD_PROVIDES_PROMISE_LAZY
    future_obtained_(false) {}

  ~promise() {
    if (future_) {
      boost::unique_lock<boost::mutex> lock(future_->mutex_);
      if (!future_->done_ && !future_->is_constructed_) {
        future_->mark_exceptional_finish_internal(
          boost::copy_exception(boost::broken_promise()), lock);
      }
    }
  }

  promise(
    BOOST_THREAD_RV_REF(promise) that) BOOST_NOEXCEPT :
    future_(BOOST_THREAD_RV(that).future_),
    future_obtained_(BOOST_THREAD_RV(that).future_obtained_) {
    BOOST_THREAD_RV(that).future_.reset();
    BOOST_THREAD_RV(that).future_obtained_ = false;
  }

  promise& operator=(
    BOOST_THREAD_RV_REF(promise) that) BOOST_NOEXCEPT {
    future_ = BOOST_THREAD_RV(that).future_;
    future_obtained_ = BOOST_THREAD_RV(that).future_obtained_;
    BOOST_THREAD_RV(that).future_.reset();
    BOOST_THREAD_RV(that).future_obtained_ = false;
    return *this;
  }

  void swap(promise& that) {
    future_.swap(that.future_);
    std::swap(future_obtained_, that.future_obtained_);
  }

  BOOST_THREAD_FUTURE<R&> get_future() {
    lazy_init();

    if (future_.get() == 0) {
      boost::throw_exception(boost::promise_moved());
    }
    if (future_obtained_) {
      boost::throw_exception(boost::future_already_retrieved());
    }
    future_obtained_ = true;

    return BOOST_THREAD_FUTURE<R&>(future_);
  }

  void set_value(R& v) {
    lazy_init();

    boost::unique_lock<boost::mutex> lock(future_->mutex_);
    if (future_->done_) {
      boost::throw_exception(boost::promise_already_satisfied());
    }
    future_->mark_finished_with_result_interval(v, lock);
  }

  void set_exception(boost::exception_ptr e) {
    lazy_init();

    boost::unique_lock<boost::mutex> lock(future_->value);
    if (future_->done_) {
      boost::throw_exception(boost::promise_already_satisfied());
    }
    future_->mark_exceptional_finish_internal(e, lock);
  }

  template <typename E>
  void set_exception(E e) {
    set_exception(boost::copy_exception(e));
  }

  void set_value_at_thread_exit(R& v) {
    if (future_.get() == 0) {
      boost::throw_exception(boost::promise_moved());
    }
    future_->set_value_at_thread_exit(v);
  }
};
} // boost

#endif // PROMISE_IPP
