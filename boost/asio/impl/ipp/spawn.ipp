#include "../hpp/async_result.hpp"
#include "../hpp/handler_cont_helpers.hpp"
// #include "../hpp/handler_type.hpp"

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/atomic_count.hpp>
#include <boost/asio/detail/handler_alloc_helpers.hpp>
// #include <boost/asio/detail/handler_cont_helpers.hpp>
#include <boost/asio/detail/noncopyable.hpp>
#include <boost/asio/detail/shared_ptr.hpp>
// #include <boost/asio/handler_type.hpp>

template <typename Handler>
basic_yield_context<Handler>::basic_yield_context(
  const boost::asio::detail::weak_ptr<callee_type>& coro,
  caller_type& ca, Handler& handler) 
  : coro_(coro), ca_(ca), handler_(handler), ec_(0) {}
