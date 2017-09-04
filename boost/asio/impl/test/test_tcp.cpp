#include <boost/asio/ip/tcp.hpp>

#include <boost/asio/io_service.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>

#include <cassert>
#include <cstring>

#include "async_result.hpp"
#include "gettable_socket_option.hpp"
#include "io_control_command.hpp"
#include "settable_socket_option.hpp"

#if defined(BOOST_ASIO_HAS_BOOST_ARRAY)
#include <boost/array.hpp>
#else
#include <array>
#endif

#if defined(BOOST_ASIO_HAS_BOOST_BIND)
#include <boost/bind.hpp>
#else
#include <functional>
#endif

void test_1() {
  try {
    boost::asio::io_service ios;
    boost::asio::ip::tcp::socket s1(ios);
    boost::asio::ip::tcp::no_delay no_delay1(true);
    s1.set_option(no_delay1);
    boost::asio::ip::tcp::no_delay no_delay2;
    s1.get_option(no_delay2);
    no_delay1 = true;
    (void)static_cast<bool>(no_delay1);
    (void)static_cast<bool>(!no_delay1);
    (void)static_cast<bool>(no_delay1.value());
  } catch (std::exception&) {}
}

void test_2() {
  boost::asio::io_service ios;
  boost::asio::ip::tcp::socket s1(ios, boost::asio::ip::tcp::v4());
  boost::system::error_code ec;

  boost::asio::ip::tcp::no_delay no_delay1(true);
  assert(no_delay1.value());
  assert(static_cast<bool>(no_delay1));
  s1.set_option(no_delay1, ec);
  assert(!ec);

  boost::asio::ip::tcp::no_delay no_delay2;
  s1.get_option(no_delay2, ec);
  assert(!ec);
  assert(no_delay2.value());
  assert(static_cast<bool>(no_delay2));
  assert(!!no_delay2);

  boost::asio::ip::tcp::no_delay no_delay3(false);
  assert(!no_delay3.value());
  assert(!static_cast<bool>(no_delay3));
  assert(!no_delay3);
  s1.set_option(no_delay3, ec);
  assert(!ec);

  boost::asio::ip::tcp::no_delay no_delay4;
  s1.get_option(no_delay4, ec);
  assert(!ec);
  assert(!no_delay4.value());
  assert(!static_cast<bool>(no_delay4));
  assert(!no_delay4);

}

namespace {
void connect_handler(const boost::system::error_code&) {}
void send_handler(const boost::system::error_code&, std::size_t) {}
void receive_handler(const boost::system::error_code&, std::size_t) {}
void write_some_handler(const boost::system::error_code&, std::size_t) {}
void read_some_handler(const boost::system::error_code&, std::size_t) {}
} // namespace

void test_3() {
#if defined(BOOST_ASIO_HAS_BOOST_ARRAY)
  using boost::array;
#else
  using std::array;
#endif
  try {
    boost::asio::io_service ios;
    char mutable_char_buffer[128] = "";
    const char const_char_buffer[128] = "";
    array<boost::asio::mutable_buffer, 2> mutable_buffers = {{
      boost::asio::buffer(mutable_char_buffer, 10),
      boost::asio::buffer(mutable_char_buffer + 10, 10)
    }};
    array<boost::asio::const_buffer, 2> const_buffers = {{
      boost::asio::buffer(const_char_buffer, 10),
      boost::asio::buffer(const_char_buffer + 10, 10)
    }};
    boost::asio::socket_base::message_flags in_flags = 0;
    settable_socket_option<void> settable_socket_option1;
    settable_socket_option<int> settable_socket_option2;
    settable_socket_option<double> settable_socket_option3;
    gettable_socket_option<void> gettable_socket_option1;
    gettable_socket_option<int> gettable_socket_option2;
    gettable_socket_option<double> gettable_socket_option3;
    io_control_command io_control_command;
    lazy_handler lazy;
    boost::system::error_code ec;
    boost::asio::ip::tcp::socket s1(ios);
    boost::asio::ip::tcp::socket s2(ios, boost::asio::ip::tcp::v4());s2.close();
    boost::asio::ip::tcp::socket s3(ios, boost::asio::ip::tcp::v6());
    boost::asio::ip::tcp::socket s4(ios,
      boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), 0));
    boost::asio::ip::tcp::socket s5(ios,
      boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v6(), 0));
#if !defined(BOOST_ASIO_WINDOWS_RUNTIME)
    boost::asio::ip::tcp::socket::native_handle_type native_socket1 =
      ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    boost::asio::ip::tcp::socket s6(
      ios, boost::asio::ip::tcp::v4(), native_socket1);
#endif
#if defined(BOOST_ASIO_HAS_MOVE)
    boost::asio::ip::tcp::socket s7(std::move(s5));
#endif
#if defined(BOOST_ASIO_HAS_MOVE)
    s1 = boost::asio::ip::tcp::socket(ios);
    s1 = std::move(s2);
#endif
    boost::asio::io_service& ios_ref = s1.get_io_service();
    (void)ios_ref;
    s1.open(boost::asio::ip::tcp::v4());
    s1.open(boost::asio::ip::tcp::v6());
    s1.open(boost::asio::ip::tcp::v4(), ec);
    s1.open(boost::asio::ip::tcp::v6(), ec);
#if !defined(BOOST_ASIO_WINDOWS_RUNTIME)
    boost::asio::ip::tcp::socket::native_handle_type native_socket2 =
      ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    s1.assign(
      boost::asio::ip::tcp::v4(), native_socket2);
    boost::asio::ip::tcp::socket::native_handle_type native_socket3 =
      ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    s1.assign(
      boost::asio::ip::tcp::v4(), native_socket3, ec);
#endif
    bool is_open = s1.is_open();
    (void)is_open;
    s1.close();
    s1.close(ec);
    boost::asio::ip::tcp::socket::native_type native_socket4 = s1.native();
    (void)native_socket4;
    boost::asio::ip::tcp::socket::native_type native_socket5 =
      s1.native_handle();
    (void)native_socket5;
    bool at_mark1 = s1.at_mark();
    (void)at_mark1;
    bool at_mark2 = s1.at_mark(ec);
    (void)at_mark2;
    std::size_t available1 = s1.available();
    (void)available1;
    std::size_t available2 = s1.available(ec);
    (void)available2;
    s1.bind(boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), 0));
    s1.bind(boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v6(), 0));
    s1.bind(boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), 0), ec);
    s1.bind(boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v6(), 0), ec);
    s1.connect(boost::asio::ip::tcp::endpoint(
      boost::asio::ip::tcp::v4(), 0));
    s1.connect(boost::asio::ip::tcp::endpoint(
      boost::asio::ip::tcp::v6(), 0));
    s1.connect(boost::asio::ip::tcp::endpoint(
      boost::asio::ip::tcp::v4(), 0), ec);
    s1.connect(boost::asio::ip::tcp::endpoint(
      boost::asio::ip::tcp::v6(), 0), ec);
    s1.async_connect(boost::asio::ip::tcp::endpoint(
      boost::asio::ip::tcp::v4(), 0), &connect_handler);
    s1.async_connect(boost::asio::ip::tcp::endpoint(
      boost::asio::ip::tcp::v6(), 0), &connect_handler);
  } catch (std::exception&) {}
}

auto main() -> decltype(0) {
  test_1(); test_2(); test_3();
  return 0;
}
