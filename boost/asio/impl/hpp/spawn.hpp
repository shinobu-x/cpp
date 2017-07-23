#include <boost/asio/detail/config.hpp>
#include <boost/coroutine/all.hpp>
#include <boost/asio/detail/weak_ptr.hpp>
#include <boost/asio/detail/wrapped_handler.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/strand.hpp>

template <typename Handler>
class basic_yield_context {
public:
  typedef boost::coroutines::coroutine<void()> callee_type;     
  typedef boost::coroutines::coroutine<void()>::caller_type caller_type;

  basic_yield_context(const boost::asio::detail::weak_ptr<callee_type>&,
    caller_type&, Handler&);

  basic_yield_context operator[] (boost::system::error_code& ec) const {
    basic_yield_context tmp(*this);
    tmp.ec_ = &ec;
    return tmp;
  }
private:
  boost::asio::detail::weak_ptr<callee_type> coro_;
  caller_type& ca_;
  Handler& handler_;
  boost::system::error_code* ec_;
};

#pragma once
#include "../ipp/spawn.ipp"
