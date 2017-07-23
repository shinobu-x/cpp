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
