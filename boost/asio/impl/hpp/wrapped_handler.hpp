#include "handler_cont_helpers.hpp"
#include <boost/asio/detail/bind_handler.hpp>
#include <boost/asio/detail/handler_alloc_helpers.hpp>
// #include <boost/asio/detail/handler_cont_helpers.hpp>
#include <boost/asio/detail/handler_invoke_helpers.hpp>

struct is_continuation_delegated {
  template <typename Dispatcher, typename Handler>
  bool operator() (Dispatcher&, Handler& handler) const {
    return boost_asio_handler_cont_helpers::is_continuation(handler);
  }
};

struct is_continuation_if_running {
  template <typename Dispatcher, typename Handler>
  bool operator() (Dispatcher& dispatcher, Handler&) const {
    return dispatcher.running_in_this_thread();
  }
};

template <typename Dispatcher, typename Handler,
  typename IsContinuation = is_continuation_delegated>
class wrapped_handler {
public:
  typedef void result_type;

  wrapped_handler(Dispatcher dispatcher, Handler& handler)
    : dispatcher_(dispatcher),
      handler_(BOOST_ASIO_MOVE_CAST(Handler)(handler)) {}

#if defined(BOOST_ASIO_HAS_MOVE)
  wrapped_handler(const wrapped_handler& other)
    : dispatcher_(other.dispatcher_),
      handler_(other.handler_) {}

  wrapped_handler(wrapped_handler&& other)
    : dispatcher_(other.dispathcer_),
      handler_(BOOST_ASIO_MOVE_CAST(Handler)(other.handler_)) {}
#endif

  void operator() () {
    dispatcher_.dispatch(BOOST_ASIO_MOVE_CAST(Handler)(handler_));
  }

  void operator() () const {
    dispatcher_.dispatch(handler_);
  }

  template <typename Arg1>
  void operator() (const Arg1 arg1) {
    dispatcher_.dispatch(boost::asio::detail::bind_handler(handler_, arg1));
  }

  template <typename Arg1>
  void operator() (const Arg1& arg1) const {
    dispatcher_.dispatch(boost::asio::detail::bind_handler(handler_, arg1));
  }

  template <typename Arg1, typename Arg2>
  void operator() (const Arg1& arg1, const Arg2& arg2) {
    dispatcher_.dispatch(
      boost::asio::detail::bind_handler(handler_, arg1, arg2));
  }

  template <typename Arg1, typename Arg2>
  void operator() (const Arg1& arg1, const Arg2& arg2) const {
    dispatcher_.dispatch(
      boost::asio::detail::bind_handler(handler_, arg1, arg2));
  }

  template <typename Arg1, typename Arg2, typename Arg3>
  void operator() (const Arg1& arg1, const Arg2& arg2, const Arg3& arg3) {
    dispatcher_.dispatch(
      boost::asio::detail::bind_handler(handler_, arg1, arg2, arg3));
  }

  template <typename Arg1, typename Arg2, typename Arg3>
  void operator() (const Arg1& arg1, const Arg2& arg2, const Arg3& arg3) const {
    dispatcher_.dispatch(
      boost::asio::detail::bind_handler(handler_, arg1, arg2, arg3));
  }

  template <typename Arg1, typename Arg2, typename Arg3, typename Arg4>
  void operator() (const Arg1& arg1, const Arg2& arg2, const Arg3& arg3,
    const Arg4& arg4) {
    dispatcher_.dispatch(
      boost::asio::detail::bind_handler(handler_, arg1, arg2, arg3, arg4));
  }

  template <typename Arg1, typename Arg2, typename Arg3, typename Arg4,
    typename Arg5>
  void operator() (const Arg1& arg1, const Arg2& arg2, const Arg3& arg3,
    const Arg4& arg4, const Arg5& arg5) {
    dispatcher_.dispatch(
      boost::asio::detail::bind_handler(
        handler_, arg1, arg2, arg3, arg4, arg5));
  }

  template <typename Arg1, typename Arg2, typename Arg3, typename Arg4,
    typename Arg5>
  void operator() (const Arg1& arg1, const Arg2& arg2, const Arg3& arg3,
    const Arg4& arg4, const Arg5& arg5) const {
    dispatcher_.dispatch(
      boost::asio::detail::bind_handler(
        handler_, arg1, arg2, arg3, arg4, arg5));
  }

// private:
  Dispatcher dispatcher_;
  Handler handler_;
};
