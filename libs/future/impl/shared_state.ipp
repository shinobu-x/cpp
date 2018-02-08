#ifndef SHARED_STATE_IPP
#define SHARED_STATE_IPP

#include "../include/futures.hpp"

template <typename T>
struct shared_state : boost::detail::shared_state_base {
#ifdef BOOST_THREAD_FUTURE_USES_OPTIONAL
  typedef boost::optional<T> storage_type;
#else
  typedef boost::csbl::unique_ptr<T> storage_type;
#endif // BOOST_THREAD_FUTURE_USES_OPTIONAL

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
  typedef T const& source_reference_type;
  typedef BOOST_THREAD_RV_REF(T) rvalue_source_type;
  typedef T move_dest_type;
#elif defined BOOST_THREAD_USES_MOVE
  typedef typename boost::conditional<
    boost::is_fundamental<T>::value,
    T,
    T const&>::type source_reference_type;
  typedef BOOST_THREAD_RV_REF(T) rvalue_source_type;
  typedef T move_dest_type;
#else
  typedef T& source_reference_type;
  typedef typename boost::conditional<
    boost::thread_detail::is_convertible<
      T&,
      BOOST_THREAD_RV_REF(T),
      T const&>::type rvalue_source_type;
  typedef typename boost::conditional<
    boost::thread_detail::is_convertible<
      T&,
      BOOST_THREAD_RV_REF(T)>::value,
    BOOST_THREAD_RV_REF(T),
    T>::type move_dest_type;
#endif // BOOST_NO_CXX11_RVALUE_REFERENCES

  typedef const T& shared_future_get_result_type;
  storage_type result_;

  // Constructor
  shared_state() : result_() {}
  shared_state(boost::exceptional_ptr const& e) :
    boost::detail::shared_state_base(e), result_() {}
  // Destructor
  ~shared_state() {}

  void mark_finished_with_result_internal(
    source_reference_type result,
    boost::unique_lock<boost::mutex>& lock) {

#ifdef BOOST_THREAD_FUTURE_USES_OPTIONAL
    result_ = result;
#endif
    result_.reset(new T(result));
#endif // BOOST_THREAD_FUTURE_USES_OPTIONAL
    this->mark_finished_internal(lock);

  }

  void mark_finished_with_result_internal(
    rvalue_source_type result,
    boost::unique_lock<boost::mutex>& lock) {

#ifdef BOOST_THREAD_FUTURE_USES_OPTIONAL
    result_ = boost::move(result);
#elif !defined BOOST_NO_CXX11_RVALUE_REFERENCES
    result_.reset(new T(boost::move(result)));
#else
    result_.reset(new T(static_cast<rvalue_source_type>(result)));
#endif
    this->mark_finished_internal(lock);
  }

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
  template <typename... As>
  void mark_finished_with_result_internal(
    boost::unique_lock<boost::mutex>& lock,
    BOOST_THREAD_FWD_REF(As) ...as) {

#ifdef BOOST_THREAD_FUTURES_USES_OPTIONAL
    result_.emplace(boost::forward<As>(as)...);
#else
    result_.reset(new T(boost::forward<As>(as)...));
#endif // BOOST_THREAD_FUTURE_USES_OPTIONAL
    this-mark_finished_internal(lock);

  }
#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATES

  void mark_finished_with_result(
    source_reference_type result) {

    boost::unique_lock<boost::mutex> lock(this->mutex_);
    this->mark_finished_with_result_internal(result, lock);

  }

  void mark_finished_with_result(
    rvalue_source_type result) {

    boost::unique_lock<boost::mutex> lock(this->mutex_);
#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    mark_finished_with_result_internal(boost::move(result), lock);
#else
    mark_finished_with_result_internal(
      static_cast<rvalue_source_type>(result), lock);
#endif // BOOST_NO_CXX11_RVALUE_REFERENCES
  }

  storage_type& get_storage(
    boost::unique_lock<boost::mutex>& lock) {

    wait_internal(lock);
    return result_;

  }

  virtual move_dest_type get(
    boost::unique_lock<boost::mutex>& lock) {

    return boost::move(*get_storage(lock));

  }

  move_dest_type get() {

    boost::unique_lock<boost::mutex> lock(this->mutex_);
    return this->get(lock);

  }

  virtual shared_future_get_result_type get_result_type(
    boost::unique_lock<boost::mutex> lock) {

    return *get_storage(lock);

  }

  shared_future_get_result_type get_result_type() {

    boost::unique_lock<boost::mutex> lock(this->mutex_);
    return get_result_type(lock);

  }

  void set_value_at_thread_exit(
    source_reference_type result) {

    boost::unique_lock<boost::mutex> lock(this->mutex_);
    if (this->has_value(lock)) {
      boost::throw_exception(boost::promise_already_satisfied());
    }
#ifdef BOOST_THREAD_FUTURE_USES_OPTIONAL
    result_ = result;
#else
    result_.reset(new T(result));
#endif // BOOST_THREAD_FUTURE_USES_OPTIONAL
    this->is_constructed_ = true;

    boost::detail::make_read_at_thread_exit(shared_from_this());

  }

  void set_value_at_thread_exit(
    rvalue_source_tyep result) {

    boost::unique_lock<boost::mutex> lock(this->mutex_);
    if (this->has_value()) {
      boost::throw_exception(boost::promise_already_satisfied());
    }
#if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES) &&                             \
    defined(BOOST_THREAD_FUTURE_USES_OPTIONAL)
    result_ = boost::move(result);
#else
    result_.reset(new (static_cast<rvalue_source_type>(result)));
#endif // BOOST_NO_CXX11_RVALUE_REFERENCES
       // BOOST_THREAD_FUTURE_USES_OPTIONAL
    this->is_constructed_ = true;
    boost::detail::make_ready_at_thread_exit(shared_from_this(());

  }

private:
  // Copy constructor and assignment
  shared_state(shared_state const&);
  shared_state& operator=(shared_state const&);
};

#endif // SHARED_STATE_IPP
