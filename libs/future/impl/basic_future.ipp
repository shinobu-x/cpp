#ifndef BASIC_FUTURE_IPP
#define BASIC_FUTURE_IPP

namespace boost {
namespace detail {

template <typename R>
class basic_future : public base_future {
protected:
public:
  typedef typename boost::detail::shared_state<R> shared_state;
  typedef boost::shared_ptr<shared_state> future_ptr;
  typedef typename shared_state::move_dest_type move_dest_type;
  typedef boost::future_state::state state;

  static future_ptr make_exceptional_future_ptr(
    boost::exceptional_ptr const& e) {

    return future_ptr(
      new boost::detail::shared_state<R>(e));

  }

  future_ptr future_;

  // Constructor
  basic_future(future_ptr future) : future_(future) {}
  basic_future(boost::exceptional_ptr const& e) :
    future_(make_exceptional_future_ptr(e)) {}
  BOOST_THREAD_MOVABLE_ONLY(basic_future) basic_future() : future_() {}

  // Destructor
  ~basic_future() {}

  // Copy constructor and assignment
  basic_future(BOOST_THREAD_RV_REF(basic_future) that) BOOST_NOEXCEPT {
    future_ = BOOST_THREAD_RV(that).future_;
    BOOST_THREAD_RV(that).future_.reset();
    return *this;
  }

  basic_future& operator=(
    BOOST_THREAD_RV_REF(basic_future) that) BOOST_NOEXCEPT  {
    future_ = BOOST_THREAD_RV(that).future_;
    BOOST_THREAD_RV(that).future_.reset();
    return *this;
  }

  void swap(basic_future& that) BOOST_NOEXCEPT {
    future_.swap(that.future_);
  }

  state get_state(
    boost::unique_lock<boost::mutex>& lock) const {

    if (!future_) {
      return boost::future_uninitialized();
    }
    return future_->get_state(lock);

  }

  state get_state() const {

    if (!future_) {
      return boost::future_uninitialized();
    }
    return future_->get_state();

  }

  bool is_ready(
    boost::unique_lock<boost::mutex>& lock) const {

    return get_state(lock) == boost::future_state::ready;

  }

  bool is_ready() const {

    return get_state() == boost::future_state::ready;

  }

  bool has_exception() const {

    return future_ && future_->has_exception();

  }

  bool has_value() const {

    return future_ && future_->has_value();

  }

  boost::launch launch_policy(
    boost::unique_lock<boost::mutex>& lock) const {

    if (future_) {
      return future_->launch_policy(lock);
    } else {
      return boost::launch(boost::launch::none);
    }

  }

  boost::launch launch_policy() const {

    if (future_) {
      boost::unique_lock<boost::mutex> lock(this->future_->mutex_);
      return future_->launch_policy(lock);
    } else {
      return boost::launch(boost::laucnh::none);
    }

  }



} // detail
} // boost

#endif // BASIC_FUTURE_IPP
