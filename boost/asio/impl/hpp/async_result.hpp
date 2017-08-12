#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/push_options.hpp>

#include "handler_type.hpp"

namespace boost {
namespace asio {

template <typename Handler>
class async_result {
public:
  typedef void type;

  explicit async_result(Handler&) {}

  type get() {}
};

namespace detail {

template <typename Handler, typename Signature>
struct async_result_init {
  explicit async_result_init(BOOST_ASIO_MOVE_ARG(Handler) orig_handler)
    : handler(BOOST_ASIO_MOVE_CAST(Handler)(orig_handler)),
      result(handler) {}

  typename handler_type<Handler, Signature>::type handler;
  async_result<typename handler_type<Handler, Signature>::type > result;
};

template <typename Handler, typename Signature>
struct async_result_type_helper {
  typedef typename async_result<
    typename handler_type<Handler, Signature>::type >::type type;
};

} // namespace detail
} // namespace asio
} // namespace boost

#include <boost/asio/detail/pop_options.hpp>

#define BOOST_ASIO_INITFN_RESULT_TYPE(h, sig) \
  typename ::boost::asio::async_result< \
    typename ::boost::asio::handler_type<h, sig>::type >::type
