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
  boost::asio::io_service ios;
  char mutable_char_buffer[128] = "";
  const char const_char_buffer[128] = "";
}

auto main() -> decltype(0) {
  test_1(); test_2();
  return 0;
}
