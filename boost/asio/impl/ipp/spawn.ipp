#include "../hpp/spawn.hpp"

template <typename Handler>
basic_yield_context<Handler>::basic_yield_context(
  const boost::asio::detail::weak_ptr<callee_type>& coro,
  caller_type& ca, Handler& handler) 
  : coro_(coro), ca_(ca), handler_(handler), ec_(0) {}
