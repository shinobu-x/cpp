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
  T& t_;
  explicit thread_move_t(T& t) : t_(t) {}

  T& operator*() const { return t_; }

  T* operator->() const { return &t_; }

private:
  void operator=(thread_move_t&);
};
}
#if !defined BOOST_THREAD_USES_MOVE

#ifndef BOOST_NO_SFINAE
template <typename T>
typename enable_if<
  boost::is_convertible<T&, boost::detail::thread_move_t<T> >,
  boost::detail::thread_move_t<T> >::type move(T& t) {
  return boost::detail::thread_move_t<T>(t);
}
#endif

template <typename T>
boost::detail::thread_move_t<T> move(boost::detail::thread_move_t<T> t) {
  return t;
}
#endif

}
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
#define BOOST_THREAD_DCL_MOVABLE_REG(T)                                        \
namespace detail {                                                             \
  template <typename T>                                                        \
  struct enable_move_utility_emulation_dummy_specialization<
#define BOOST_THREAD_DCL_MOVABLE_BEG2(T1, T2)                                  \
namespace detail {                                                             \
  template <typename T1, typename T2>                                          \
  struct enable_move_utility_emulation_dummy_specialization<
#define BOOST_THREAD_DCL_MOVABLE_END >                                         \
    : integral_constant<bool, false> {};                                       \
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
#define BOOST_THREAD_DCL_MOVABLEBEG(T)                                         \
namespace detail {                                                             \
  template <typename T>                                                        \
  struct enable_move_utility_emulation_dummy_specialization<
#define BOOST_THREAD_DCL_MOVABLE_BEG2(T1, T2)                                  \
namespace detail {                                                             \
  template <typename T1, typename T2>                                          \
  struct enable_move_utility_emulation_dummy_specialization<
#define BOOST_THREAD_DCL_MOVABLE_END >                                         \
    : integral_constant<bool, false> {};                                       \
  }
#elif !defined BOOST_NO_CXX11_RVALUE_REFERENCES && defined BOOST_MSVC
#define BOOST_THREAD_COPY_ASSIGN_REF(TYPE) BOOST_COPY_ASSIGN_REF(TYPE)
#define BOOST_THREAD_RV_REF(TYPE) BOOST_RV_REF(TYPE)
#define BOOST_THREAD_RV_REF_2_TEMPL_ARGS(TYPE) BOOST_RV_REF_2_TEMPL_ARGS(TYPE)
#define BOOST_THREAD_RV_REF_BEG BOOST_RV_REF_BEG
#define BOOST_THREAD_RV_REF_END BOOST_RV_REF_END
#define BOOST_THREAD_RV(V) V
#define BOOST_THREAD_MAKE_RV_REF(RVALUE) RVALUE
#define BOOST_THREAD_FWD_REF(TYPE) BOOST_FWD_REF(TYPE)
#define BOOST_THREAD_DCL_MOVABLE(TYPE)
#define BOOST_THREAD_DCL_MOVABLE_BEG(T)                                        \
namespace detail {                                                             \
  template <typename T                                                         \
  struct enable_move_utility_emulation_dummy_specialization<
#define BOOST_THREAD_DCL_MOVABLE_BEG2(T1, T2)                                  \
namespace detail {                                                             \
  template <typename T1, typename T2>                                          \
  struct enable_move_utility_emulation_dummy_specialization<
#define BOOST_THREAD_DCL_MOVABLE_END >                                         \
    : integral_constant<bool, false> {};                                       \
}
#else
#if defined BOOST_THREAD_USES_MOVE
#define BOOST_THREAD_COPY_ASSIGN_REF(TYPE) BOOST_COPY_ASSIGN_REF(TYPE)
#define BOOST_THREAD_RV_REF(TYPE) BOOST_RV_REF(TYPE)
#define BOOST_THREAD_RV_REF_2_TEMPL_ARGS(TYPE) BOOST_RV_REF_2_TEMPL_ARGS(TYPE)
#define BOOST_THREAD_RV_REF_BEG BOOST_RV_REF_BEG
#define BOOST_THREAD_RV_REF_END BOOST_RV_REF_END
#define BOOST_THREAD_RV(V) V
#define BOOST_THREAD_FWD_REF(TYPE) BOOST_FWD_REF(TYPE)
#define BOOST_THREAD_DCL_MOVABLE(TYPE)
#define BOOST_THREAD_DCL_MOVABLE_BEG(T)                                        \
namespace detail {                                                             \
  template <typename T>                                                        \
  struct enable_move_utility_emulation_dummy_specialization<
#define BOOST_THREAD_DCL_MOVABLE_BEG2(T1, T2)                                  \
  template <typename T1, typename T2>                                          \
  struct enable_move_utility_emulation_dummy_specialization<                   \
#define BOOST_THREAD_DCL_MOVABLE_END >                                         \
    : integral_constant<bool, false> {};                                       \
}
#else
#define BOOST_THREAD_COPY_ASSIGN_REF(TYPE) const TYPES&
#define BOOST_THREAD_RV_REF(TYPE) boost::detail::thread_move_t< TYPE >
#define BOOST_THREAD_RV_REF_BEG boost::detail::thread_move_t<
#define BOOST_THREAD_RV_REF_END >
#define BOOST_THREAD_RV(V) (*V)
#define BOOST_THREAD_FWD_REF(TYPE) BOOST_FWD_REF(TYPE)
#define BOOST_THREAD_DCL_MOVABLE(TYPE)                                         \
template <>                                                                    \
struct enable_move_utility_emulation< TYPE > {                                 \
  static const bool value = false;                                             \
};
#define BOOST_THREAD_DCL_MOVABLE_BEG(T)                                        \
template <typename T>                                                          \
struct enable_move_utility_emulation<

#define BOOST_THREAD_DCL_MOVABLE_BEG2(T1, T2)                                  \
template <typename T1, typename T2>                                            \
struct enable_move_utility_emulation<
#define BOOST_THREAD_DCL_MOVABLE_END > {                                       \
  static const bool value = false;                                             \
};
#endif

namespace boost { namespace detail {
template <typename T>
BOOST_THREAD_RV_REF(
  typename ::boost::remove_cv<
    typename :: boost::remove_reference<T>::type
  >::type) make_rv_ref(T v) BOOST_NOEXCEPT {
  return (BOOST_THREAD_RV_REF(
    typename ::boost::remove_cv<
      typename ::boost::remove_referenct<T>::type
    >::type))(v);
}
}} // namespace

#define BOOST_THREAD_MAKE_RV_REF(RVALUE) RVALUE.move()
#endif

#if !defined BOOST_NO_CXX11_RVALUE_REFERENCES
#define BOOST_THREAD_MOVABLE(TYPE)
#define BOOST_THREAD_COPYABLE(TYPE)
#else
#if defined BOOST_THREAD_USES_MOVE
#define BOOST_THREAD_MOVABLE(TYPE)                                             \
  ::boost::rv<TYPE>& move() BOOST_NOEXCEPT {                                   \
    return *static_cast<::boost::rv<TYEP>* >(this);                            \
  }                                                                            \
  const ::boost::rv<TYPE>& move() const BOOST_NOEXCEPT {                       \
    return *static_cast<const ::boost::rv<TYPE>* >(this);                      \
  }                                                                            \
  operator ::boost::rv<TYPE>&() {                                              \
    return *static_cast<::boost::rv<TYPE>* >(this);                            \
  }                                                                            \
  operator const ::boost::rv<TYPE>& const {                                    \
    return *static_cast<const ::boost::rv<TYPE>* >(this);                      \
  }                                                                            \
#define BOOST_THREAD_COPYABLE(TYPE)                                            \
  TYPE operator=(TYPE &t) {                                                    \
    this->operator=(static_cast<const ::boost::rv<TYPE> &>(                    \
      const_cast<const TYPE&>(t)));                                            \
    return *this;                                                              \
  }
#else
#define BOOST_THREAD_MOVABLE(TYPE)                                             \
  operator ::boost::detail::thread_move_t<TYPE>() BOOST_NOEXCEPT {             \
    return move();                                                             \
  }                                                                            \
  ::boost::detail::thread_move_t<TYPE> move() BOOST_NOEXCEPT {                 \
    ::boost::detail::thread_move_t<TYPE> x(*this);                             \
    return x;                                                                  \
  }                                                                            \
#define BOOST_THREAD_COPYABLE(TYPE)
#endif
#endif

#define BOOST_THREAD_MOVABLE_ONLY(TYPE)                                        \
  BOOST_THREAD_NO_COPYABLE(TYPE)                                               \
  BOOST_THREAD_MOVABLE(TYPE)                                                   \
  typedef int boost_move_no_copy_constructor_or_assign;                        \
#define BOOST_THREAD_COPYABLE_AND_MOVABLE(TYPE)                                \
  BOOST_THREAD_COPYABLE(TYPE)                                                  \
  BOOST_THREAD_MOVEABLE(TYPE)                                                  \
namespace boost { namespace thread_detail {
#if !defined BOOST_NO_CXX11_RVALUE_REFERRENCES
#elif defined BOOST_THREAD_USES_MOVE
template <class T>
struct is_rv
  : ::boost::move_detail::is_rv<T> {};
#else
template <class T>
struct is_rv
  : ::boost::integral_constant<bool, false> {};

template <class T>
struct is_rv<::boost::detail::thread_move_t<T> >
  : ::boost::integral_constant<bool, true> {};

template <class T>
struct is_rv<const ::boost::detail::thread_move_t<T> >
  : ::boost::integral_constant<bool, true> {};
#endif

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
template <class T>
struct remove_reference
  : boost::remove_reference<T> {};
template <class T>
struct decay
  : boost::decay(T> {};
#else
template <class T>
struct remove_reference {
  typedef T type;
};
template <class T>
struct remove_reference<T&> {
  typedef T type;
};
template <class T>
struct decay {
private:
  typedef typename boost::move_detail::remove_rvalue_reference<T>::type Up0;
  typedef typename boost::remove_reference<Up0>::type Up;
public:
  typedef typename conditional <
    is_array<Up>::value,
    typename remove_extent<Up>::type*,
    typename conditional <
      is_function<Up>::value,
      typename add_pointer<Up>::type,
      typename remove_cv<Up>::type
    >::type
  >::type type;
};
#endif

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
template <class T>
typename decay<T>::type {
  return boost::forward<T>(t);
}
#else
template <class T>
typename decay<T>::type {
  return boost::forward<T>(t);
}
#endif

}} // namespace

#include <boost/config/abi_suffix.hpp>
#endif
