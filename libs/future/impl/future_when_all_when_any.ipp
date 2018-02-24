#ifndef FUTURE_WHEN_ALL_WHEN_ANY_IPP
#define FUTURE_WHEN_ALL_WHEN_ANY_IPP

#include <include/futures.hpp>

namespace boost {
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
  boost::detail::future_async_shared_state_base<
    boost::csbl::vector<F> > {
  typedef boost::csbl::vector<F> vector_type;
  typedef typename F::value_type value_type;
  vector_type v_;

  static void run(boost::shared_ptr<boost::detail::shared_state_base> that) {
    future_when_all_vector_shared_state* that_ =
      static_cast<future_when_all_vector_shared_state*>(that.get());

    try {
      boost::wait_for_all(
        that_->v_.begin(), that_->v_.end());
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
    this->shared_from_this());
#else // BOOST_THREAD_FUTURE_BLOCKING
  boost::thread(
    &future_when_all_vector_shared_state::run,
    this->shared_from_this()).detach();
#endif // BOOST_THREAD_FUTURE_BLOCKING
  }

  template <typename InputIter>
  future_when_all_vector_shared_state(
    boost::detail::input_iterator_tag,
    InputIter begin,
    InputIter end) :
    v_(
      std::make_move_iterator(begin),
      std::make_move_iterator(end)) {}

  future_when_all_vector_shared_state(
    vector_tag,
    BOOST_THREAD_RV_REF(boost::csbl::vector<F>) v) :
    v_(boost::move(v)) {}

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
  template <typename T, typename... Ts>
  future_when_all_vector_shared_state(
    values_tag,
    BOOST_THREAD_FWD_REF(T) f,
    BOOST_THREAD_FWD_REF(Ts) ...fs) {
    v_.push_back(boost::forward<T>(f));

    typename alias_t<char[]>::type {
      (
        v_.push_back(boost::forward<Ts>(fs)),
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
  boost::detail::future_async_shared_state_base<
    boost::csbl::vector<F> > {
  typedef boost::csbl::vector<F> vector_type;
  typedef typename F::value_type value_type;
  vector_type v_;

  static void run(boost::shared_ptr<boost::detail::shared_state_base> that) {
    future_when_any_vector_shared_state* that_ =
      static_cast<future_when_any_vector_shared_state*>(that.get());

    try {
      boost::wait_for_any(
        that_->v_.begin(), that_->v_.end());
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
    this->thr_ = boost::thread(
      &future_when_any_vector_shared_state::run,
      this->shared_from_this());
#else // BOOST_THREAD_BLOCKING
    boost::thread(
      &future_when_any_vector_shared_state::run,
      this->shared_from_this()).detach();
#endif // BOOST_THREAD_FUTURE_BLOCKING
  }

  template <typename InputIter>
  future_when_any_vector_shared_state(
    boost::detail::input_iterator_tag,
    InputIter begin,
    InputIter end) :
    v_(
       std::make_move_iterator(begin),
       std::make_move_iterator(end)) {}

  future_when_any_vector_shared_state(
    vector_tag,
    BOOST_THREAD_RV_REF(boost::csbl::vector<F>) v) :
    v_(boost::move(v)) {}

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
  template <typename T, typename... Ts>
  future_when_any_vector_shared_state(
    values_tag,
    BOOST_THREAD_FWD_REF(T) f,
    BOOST_THREAD_FWD_REF(Ts) ...fs) {
    v_.push_back(boost::forward<T>(f));

    typename alias_t<char[]>::type {
      (
        v_.push_back(boost::forward<Ts>(fs)),
        '0'
      )...,
      '0'
    };
  }
#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATES

  ~future_when_any_vector_shared_state() {}
};

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATE
struct wait_for_all_wrapper {
  template <typename... T>
  void operator()(T&& ...v) {
    boost::wait_for_all(boost::forward<T>(v)...);
  }
};

struct wait_for_all_wrapper {
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

} // detail
#endif // BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY
} // boost
#endif // FUTURE_WHEN_ALL_WHEN_ANY_IPP
