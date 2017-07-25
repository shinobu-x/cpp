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

template <typename Handler, typename T>
class coro_handler {
public:
  coro_handler(basic_yield_context<Handler> ctx)
    : coro_(ctx.coro_.lock()),
      ca_(ctx.ca),
      handler_(ctx.handler_),
      ready_(0),
      ec_(ctx.ec_),
      value_(0) {}

  void operator() (T value) {
    *ec_ = boost::system::error_code();
    *value_ = BOOST_ASIO_MOVE_CAST(T)(value);
    if (--*ready_ == 0)
      (*coro_)();
  }

  void operator() (boost::system::error_code ec, T value) {
    *ec_ = ec;
    *value_ = BOOST_ASIO_MOVE_CAST(T)(value);
    if (--*ready_ == 0)
      (*coro_)();
  }

// private:
  boost::asio::detail::shared_ptr<
    typename basic_yield_context<Handler>::callee_type> coro_;
  typename basic_yield_context<Handler>::caller_type& ca_;
  Handler& handler_;
  boost::asio::detail::atomic_count* ready_;
  boost::system::error_code* ec_;
  T* value_;
};

template <typename Handler>
class coro_handler<Handler, void> {
public:
  coro_handler(basic_yield_context<Handler> ctx)
    : coro_(ctx.coro_.lock()),
      ca_(ctx.ca_),
      handler_(ctx.handler_),
      ready_(0),
      ec_(ctx.ec_) {}

  void operator() () {
    *ec_ = boost::system::error_code();
    if (--*ready_ == 0)
      (*coro_)();
  }

  void operator() (boost::system::error_code ec) {
    *ec_ = ec;
    if (--*ready_ == 0)
      (*coro_)();
  }

// private:
  boost::asio::detail::shared_ptr<
    typename basic_yield_context<Handler>::callee_type> coro_;
  typename basic_yield_context<Handler>::caller_type& ca_;
  Handler& handler_;
  boost::asio::detail::atomic_count* ready_;
  boost::system::error_code* ec_;
};

template <typename Handler, typename T>
inline void* asio_handler_allocate(std::size_t size,
  coro_handler<Handler, T>* this_handler) {
  return boost_asio_handler_alloc_helpers::allocate(
    size, this_handler->handler_);
}

template <typename Handler, typename T>
inline void asio_handler_deallocate(void* pointer, std::size_t size,
  coro_handler<Handler, T>* this_handler) {
  boost_asio_handler_alloc_helpers::deallocate(
    pointer, size, this_handler->handler_);
}

template <typename Handler, typename T>
inline bool asio_handler_is_continuation(coro_handler<Handler, T>*) {
  return true;
}

template <typename Function, typename Handler, typename T>
inline void asio_handler_invoke(Function& function,
  coro_handler<Handler, T>* this_handler) {
  boost_asio_handler_invoke_helpers::invoke(
    function, this_handler->handler_);
}

template <typename Function, typename Handler, typename T>
inline void asio_handler_invoke(const Function& function,
  coro_handler<Handler, T>* this_handler) {
  boost_asio_handler_invoke_helpers::invoke(
    function, this_handler->handler_);
}

template <typename Handler, typename ReturnType>
struct handler_type<basic_yield_context<Handler>, ReturnType()>
{
  typedef coro_handler<Handler, void> type;
};

template <typename Handler, typename ReturnType, typename Arg1>
struct handler_type<basic_yield_context<Handler>, ReturnType(Arg1)> {
  typedef handler_type<Handler, Arg1> type;
};

template <typename Handler, typename ReturnType>
struct handler_type<basic_yield_context<Handler>,
  ReturnType(boost::system::error_code)> {
  typedef coro_handler<Handler, void> type;
};

template <typename Handler, typename ReturnType, typename Arg2>
struct handler_type<basic_yield_context<Handler>,
  ReturnType(boost::system::error_code, Arg2)> {
  typedef coro_handler<boost::system::error_code, Arg2> type;
};

template <typename Handler, typename T>
class async_result<coro_handler<Handler, T> > {
public:
  typedef T type;

  explicit async_result(coro_handler<Handler, T>& h)
    : handler_(h),
      ca_(h.ca_),
      ready_(2) {
    h.ready_ = &ready_;
    out_ec_ = h.ec_;
    if (!out_ec_) h.ec_ = &ec_;
    h.value_ = &value_;
  }

  type get() {
    handler_.coro_.reset();
    if (--ready_ != 0)
      ca_();
    if (!out_ec_ && ec_)
      throw boost::system::system_error(ec_);
    return BOOST_ASIO_MOVE_CAST(type)(value_);
  }

private:
  coro_handler<Handler, T>& handler_;
  typename basic_yield_context<Handler>::caller_type& ca_;
  boost::asio::detail::atomic_count ready_;
  boost::system::error_code* out_ec_;
  boost::system::error_code ec_;
  type value_;
};

template <typename Handler>
class async_result<coro_handler<Handler, void> > {
public:
  typedef void type;

  explicit async_result(coro_handler<Handler, void>& h)
    : handler_(h),
      ca_(h.ca_),
      ready_(2) {
    h.ready_ = &ready_;
    out_ec_ = h.ec_;
    if (!out_ec_) h.ec_ = &ec_;
  }

  void get() {
    handler_.coro_.reset();
    if (--ready_ != 0)
      ca_();
    if (!out_ec_ && ec_)
      throw boost::system::system_error(ec_);
  }

private:
  coro_handler<Handler, void>& handler_;
  typename basic_yield_context<Handler>::caller_type& ca_;
  boost::asio::detail::atomic_count ready_;
  boost::system::error_code* out_ec_;
  boost::system::error_code ec_;
};
