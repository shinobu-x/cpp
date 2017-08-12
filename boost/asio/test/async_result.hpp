#include "../impl/hpp/async_result.hpp"
#include "../impl/hpp/handler_type.hpp"

namespace archetypes {

struct lazy_handler {};

struct concrete_handler {
  concrete_handler(lazy_handler) {}

  template <typename Arg1>
  void operator()(Arg1) {}

  template <typename Arg1, typename Arg2>
  void operator()(Arg1, Arg2) {}
};

} // namespace

namespace boost {
namespace asio {

template <typename Signature>
struct handler_type<archetypes::lazy_handler, Signature> {
  typedef archetypes::concrete_handler type;
};

template <>
class async_result<archetypes::concrete_handler> {
public:
  typedef int type;

  explicit async_result(archetypes::concrete_handler&) {}

  type get() {
    return 42;
  }
};

} // namespace asio
} // namespace boost

