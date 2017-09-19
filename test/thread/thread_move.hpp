#include <boost/thread/detail/config.hpp>
#ifndef BOOST_NO_SFINAE
#include <boost/core/enable_if.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/type_traits/remove_cv.hpp>
#include <boost/type_traits/decay.hpp>
#include <boost/type_traits/conditional.hpp>
#include <boost/type_traits/remove_extent.hpp>
#include <boost/type_traits/is_array.hpp>
#include <boost/type_traits/is_function.hpp>
#include <boost/type_traits/add_pointer.hpp>
#endif
#include <boost/thread/detail/delete.hpp>
#include <boost/move/utility.hpp>
#include <boost/move/traits.hpp>
#include <boost/config/abi_prefix.hpp>
namespace boost { namespace detail {
template <typename T>
struct enable_move_utility_emulation_dummy_specialization;

template <typename T>
struct thread_move_t {
  T& t;
  explicit thread_move_t(T& t) : t_(t) {}

  T& operator*() const { return t; }

  T* operator->() const { return &t; }

private:
  void operator=(thread_move_t&);
};

#if !defined BOOST_THREAD_USES_MOVE

#ifndef BOOST_NO_SFINAE
template <typename T>
typename enable_if<boost::is_convertible<T&, boost::detail::thread_move_t<T>,
  boost::detail::thread_move_t<T> >::type move(T& t) {
  return boost::detail::thread_move_t<T>(t);
}
#endif
template <typename T>
boost::detail::thread_move_t<T> move(boost::detail::thread_move_t<T> t) {
  return t;
}
#endif

#if !defined BOOST_NO_CXX11_RVALUE_REFERENCES
#define BOOST_THREAD_COPY_ASSIGN_REF(TYPE) BOOST_COPY_ASSIGN_REF(TYPE)
#define BOOST_THREAD_RV_REF(TYPE) BOOST_RV_REF(TYPE)
#define BOOST_THREAD_RV_REF_2_TEMPL_ARGS(TYPE) BOOST_RV_REF_2_TEMPL_ARGS(TYPE)
#define BOOST_THREAD_RV_REF_BEG BOOST_RV_REF_BEG
#define BOOST_THREAD_RV_REF_END BOOST_RV_REF_END
#define BOOST_THREAD_RV(V) V
#define BOOST_THREAD_MAKE_RV_REV(RVALUE) RVALUE
#define BOOST_THREAD_FWD_REF(TYPE) BOOST_FWD_REF(TYPE)
#define BOOST_THREAD_DCL_MOVABLE(TYPE)
#define BOOST_THREAD_DCL_MOVABLE_REG(T) \
namespace detail { \
  template <typename T> \
  struct enable_move_utility_emulation_dummy_specialization<
#define BOOST_THREAD_DCL_MOVABLE_BEG2(T1, T2) \
namespace detail { \
  template <typename T1, typename T2> \
  struct enable_move_utility_emulation_dummy_specialization<
#define BOOST_THREAD_DCL_MOVABLE_END > \
    : integral_constant<bool, false> {}; \
  }
#else
#if defined BOOST_THREAD_USES_MOVE
#define BOOST_THREAD_COPY_ASSIGN_REF(TYPE) BOOST_COPY_ASSIGN_REF(TYPE)
#define BOOST_THREAD_RV_REF(TYPE) BOOST_RV_REF(TYPE)
#define BOOST_THREAD_RV_REF_2_TEMPL_ARGS(TYPE) BOOST_RV_REF_2_TEMPL_ARGS(TYPE)
#define BOOST_THREAD_RV_REF_BEG BOOST_RV_REF_REG
#define BOOST_THREAD_RV_REF_END BOOST_RV_REF_END
#define BOOST_THREAD_RV(V) V
#define BOOST_THREAD_FWD_REF(TYPE) BOOST_FWD_REF(TYPE)
#define BOOST_THREAD_DCL_MOVABLE(TYPE)
#define BOOST_THREAD_DCL_MOVABLEBEG(T) \
namespace detail { \
  template <typename T> \
  struct enable_move_utility_emulation_dummy_specialization<
#define BOOST_THREAD_DCL_MOVABLE_BEG2(T1, T2) \
namespace detail { \
  template <typename T1, typename T2> \
  struct enable_move_utility_emulation_dummy_specialization<
#define BOOST_THREAD_DCL_MOVABLE_END > \
    : integral_constant<bool, false> {}; \
  }
#else
#define BOOST_THREAD_COPY_ASSIGN_REF(TYPE) const TYPES&
#define BOOST_THREAD_RV_REF(TYPE) boost::detail::thread_move_t< TYPE >
#define BOOST_THREAD_RV_REF_BEG boost::detail::thread_move_t<
#define BOOST_THREAD_RV_REF_END >
#define BOOST_THREAD_RV(V) (*V)
#define BOOST_THREAD_FWD_REF(TYPE) BOOST_FWD_REF(TYPE)
#define BOOST_THREAD_DCL_MOVABLE(TYPE) \
template <> \
struct enable_move_utility_emulation< TYPE > { \
  static const bool value = false; \
};
#define BOOST_THREAD_DCL_MOVABLE_BEG(T) \
template <typename T> \
struct enable_move_utility_emulation<

#define BOOST_THREAD_DCL_MOVABLE_BEG2(T1, T2) \
template <typename T1, typename T2> \
struct enable_move_utility_emulation<
#define BOOST_THREAD_DCL_MOVABLE_END > { \
  static const bool value = false; \
};
#endif

namespace boost { namespace detail {
template <typename T>
BOOST_THREAD_RV_REF(typename ::boost::remove_cv<typename :: boost::remove_reference<T>::type>::type) make_rv_ref(T v) BOOST_NOEXCEPT {
  return (BOOST_THREAD_RV_REF(typename ::boost::remove_cv<typename ::boost::remove_referenct<T>::type>::type))(v);
}
