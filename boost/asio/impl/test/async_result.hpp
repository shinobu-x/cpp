#include <boost/asio/async_result.hpp>
#include <boost/asio/handler_type.hpp>

struct lazy_handler {};

struct concrete_handler {
  concrete_handler(lazy_handler) {}

  template <typename arg1>
  void operator()(arg1) {}

  template <typename arg1, typename arg2>
  void operator()(arg1, arg2) {}
};

namespace boost { namespace asio {
template <typename signature>
struct handler_type<lazy_handler, signature> {
  typedef concrete_handler type;
};

template <>
class async_result<concrete_handler> {
public:
  typedef int type;

  explicit async_result(concrete_handler) {}

  type get() {
    return 42;
  }
};
}}
