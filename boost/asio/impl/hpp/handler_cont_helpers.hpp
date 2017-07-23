#include "addressof.hpp"
#include "handler_continuation_hook.hpp"

#include <boost/asio/detail/config.hpp>
// #include <boost/asio/detail/addressof.hpp>
// #include <boost/asio/handler_continuation_hook.hpp>

template <typename Context>
inline bool is_continuation(Context& context) {
#if !defined(BOOST_ASIO_HAS_HANDLER_HOOKS)
  return false;
#else
  using boost::asio::asio_handler_is_continuation;
  return asio_handler_is_continuation(
    boost::asio::detail::addressof(context));
#endif
}
