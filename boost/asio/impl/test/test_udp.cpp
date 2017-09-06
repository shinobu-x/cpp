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
    const char const_char_buffer[128] = "";
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
    boost::asio::ip::udp::socket s1(ios);
    boost::asio::ip::udp::socket s2(ios, boost::asio::ip::udp::v4());
    boost::asio::ip::udp::socket s3(ios,
      boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), 0));
    boost::asio::ip::udp::socket s4(ios,
      boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v6(), 0));
#if !defined(BOOST_ASIO_WINDOWS_RUNTIME)
    boost::asio::ip::udp::socket::native_handle_type native_socket1 =
      ::socket(AF_INET, SOCK_STREAM, IPPROTO_UDP);
    boost::asio::ip::udp::socket s6(ios,
      boost::asio::ip::udp::v4(), native_socket1);
#endif
#if defined(BOOST_ASIO_HAS_MOVE)
    boost::asio::ip::udp::socket s7(std::move(s6));
    s1 = boost::asio::ip::udp::socket(ios);
    s1 = std::move(s2);
#endif
    boost::asio::io_service& ios_ref = s1.get_io_service();
    (void)ios_ref;
    boost::asio::ip::udp::socket::lowest_layer_type& lowest_layer1 =
      s1.lowest_layer();
    (void)lowest_layer1;
    const boost::asio::ip::udp::socket& s8 = s1;
    const boost::asio::ip::udp::socket::lowest_layer_type& lowest_layer2 =
      s8.lowest_layer();
    (void)lowest_layer2;
    s1.open(boost::asio::ip::udp::v4());
    s1.open(boost::asio::ip::udp::v6());
    s1.open(boost::asio::ip::udp::v4(), ec);
    s1.open(boost::asio::ip::udp::v6(), ec);
#if !defined(BOOST_ASIO_WINDOWS_RUNTIME)
    boost::asio::ip::udp::socket::native_handle_type native_socket2 =
      ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    s1.assign(boost::asio::ip::udp::v4(), native_socket2);
    boost::asio::ip::udp::socket::native_handle_type native_socket3 =
      ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    s1.assign(boost::asio::ip::udp::v4(), native_socket3, ec);
#endif
    bool is_open = s1.is_open();
    (void)is_open;
    s1.close();
    s1.close(ec);
    boost::asio::ip::udp::socket::native_type native_socket4 = s1.native();
    (void)native_socket4;
    boost::asio::ip::udp::socket::native_handle_type native_socket5 =
      s1.native_handle();
    (void)native_socket5;
    s1.cancel();
    s1.cancel(ec);
    bool at_mark1 = s1.at_mark();
    (void)at_mark1;
    bool at_mark2 = s1.at_mark(ec);
    (void)at_mark2;
    std::size_t available1 = s1.available();
    (void)available1;
    std::size_t available2 = s1.available(ec);
    (void)available2;
    s1.bind(boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), 0));
    s1.bind(boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v6(), 0));
  } catch (std::exception&) {}

}

} // namespae

auto main() -> decltype(0) {
  test_1();
  return 0;
}
