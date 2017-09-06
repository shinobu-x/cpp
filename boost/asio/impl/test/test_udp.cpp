#include <boost/asio/ip/udp.hpp>

#include <boost/asio/io_service.hpp>

#include <cstring>

#include "async_result.hpp"
#include "gettable_socket_option.hpp"
#include "io_control_command.hpp"
#include "settable_socket_option.hpp"

namespace {
void connect_handler(const boost::system::error_code&) {}
void send_handler(const boost::system::error_code&, std::size_t) {}
void receive_handler(const boost::system::error_code&, std::size_t) {}

void test_1() {
  try {
    boost::asio::io_service ios;
    char mutable_char_buffer[128] = "";
    const const_char_buffer[128] = "";
    boost::asio::socket_base::message_flags in_flags = 0;
    settable_socket_option<void> settable_socket_option1;
    settable_socket_option<int> settable_socket_option2;
    settable_socket_option<double> settable_socket_option3;
    gettable_socket_option<void> gettable_socket_option1;
    gettable_socket_option<int> gettable_socket_option2;
    gettable_socket_option<double> gettable_socket_option3;
    io_control_command io_control_command;
    lazy_handler lazy;
    boost::sysem::error_code ec;
    boost::asio::ip::udp::socket s1(ios);
    boost::asio::ip::udp::socket s2(ios, boost::asio::ip::udp::v4());
    boost::asio::ip::udp::socket s3(ios,
      boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), 0));
    boost::asio::ip::udp::socket s4(ios,
      boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v6(), 0));
#if !define(BOOST_ASIO_WINDOWS_RUNTIME)
    boost::asio::ip::udp::socket::native_handle_type native_socket1 =
      ::socket(AF_INET, SOCK_STREAM, IPPROTO_UDP);
    boost::asio::ip::udp::socket s6(ios,
      boost::asio::ip::udp::v4(), native_socket1);
#endif
#if define(BOOST_ASIO_HAS_MOVE)
   boost::asio::ip::udp::socket s7(std::move(s6));
   s1 = boost::asio::ip::udp::socket(ios);
   s1 = std::move(s2);
#endif
   boost::asio::io_service& ios_ref = s1.get_io_service();

  } catch (std::exception&) {}

}

} // namespae

auto main() -> decltype(0) {
  test_1();
  return 0;
}
