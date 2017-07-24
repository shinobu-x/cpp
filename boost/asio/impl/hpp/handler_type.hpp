#include <boost/asio/detail/config.hpp>

template <typename Handler, typename Signature>
struct handler_type {
  typedef Handler type;
};


template <typename Handler, typename Signature>
struct handler_type<const Handler, Signature>
  : handler_type<Handler, Signature> {};

template <typename Handler, typename Signature>
struct handler_type<volatile Handler, Signature>
  : handler_type<Handler, Signature> {};

template <typename Handler, typename Signature>
struct handler_type<const volatile Handler, Signature>
  : handler_type<Handler, Signature> {};

template <typename Handler, typename Signature>
struct handler_type<const Handler&, Signature>
  : handler_type<Handler, Signature> {};

template <typename Handler, typename Signature>
struct handler_type<volatile Handler&, Signature>
  : handler_type<Handler, Signature> {};

template <typename Handler, typename Signature>
struct handler_type<const volatile Handler&, Signature>
  : handler_type<Handler, Signature> {};

template <typename Handler, typename Signature>
struct handler_type<Handler&, Signature>
  : handler_type<Handler, Signature> {};

#if defined(BOOST_ASIO_HAS_MOVE)
template <typename Handler, typename Signature>
struct handler_type<Handler&&, Signature>
  : handler_type<Handler, Signature> {};
#endif

template <typename ReturnType, typename Signature>
struct handler_type<ReturnType(), Signature>
  : handler_type<ReturnType(*)(), Signature> {};

template <typename ReturnType, typename Arg1, typename Signature>
struct handler_type<ReturnType(Arg1), Signature>
  : handler_type<ReturnType(*)(Arg1), Signature> {};

template <typename ReturnType, typename Arg1, typename Arg2, typename Signature>
struct handler_type<ReturnType(Arg1, Arg2), Signature>
  : handler_type<ReturnType(*)(Arg1, Arg2), Signature> {};

template <typename ReturnType, typename Arg1, typename Arg2, typename Arg3,
  typename Signature>
struct handler_type<ReturnType(Arg1, Arg2, Arg3), Signature>
  : handler_type<ReturnType(*)(Arg1, Arg2, Arg3), Signature> {};

template <typename ReturnType, typename Arg1, typename Arg2, typename Arg3,
  typename Arg4, typename Signature>
struct handler_type<ReturnType(Arg1, Arg2, Arg3, Arg4), Signature>
  : handler_type<ReturnType(*)(Arg1, Arg2, Arg3, Arg4), Signature> {};

template <typename ReturnType, typename Arg1, typename Arg2, typename Arg3,
  typename Arg4, typename Arg5, typename Signature>
struct handler_type<ReturnType(Arg1, Arg2, Arg3, Arg4, Arg5), Signature>
  : handler_type<ReturnType(*)(Arg1, Arg2, Arg3, Arg4, Arg5), Signature> {};
