#include "handler_type.hpp"

#include <boost/asio/detail/config.hpp>

template <typename Handler>
class async_result {
public:
  typedef void type;

  explicit async_result(Handler&) {}

  type get() {}
};

template <typename Handler, typename Signature>
struct async_result_init {
  explicit async_result_init(BOOST_ASIO_MOVE_ARG(Handler) orig_handler)
    : handler(BOOST_ASIO_MOVE_CAST(Handler)(orig_handler)),
      result(handler) {}

  typename handler_type<Handler, Signature>::type handler;
  async_result<typename handler_type<Handler, Signature>::type> result;
};

template <typename Handler, typename Signature>
struct async_result_type_helper {
  typedef typename async_result<
    typename handler_type<Handler, Signature>::type >::type type;
};


