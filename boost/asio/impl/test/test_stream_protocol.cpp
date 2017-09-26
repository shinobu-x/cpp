#include <boost/asio/stream_protocol.hpp>

#include <boost/asio/io_service.hpp>

#include <string>

namespace compile {

void connect_handler(const boost::system::error_code&) {}

void send_handler(const boost::system::error_code&, std::size_t) {}

void receive_handler(const boost::system::error_code&, std::size_t) {}

void write_some_handler(const boost::system::error_code&, std::size_t) {}

void read_some_handler(const boost::system::error_code&, std::size_t) {}

void test() {
  try {
    boost::asio::io_service ios;
    char mutable_char_buffer[128] = "";
    const char const_char_buffer[128] = "";
    boost::asio::socket_base::message_flags in_flags = 0;
    boost::asio::socket_base::keep_alive socket_option;
    boost::asio::socket_base::bytes_readable io_control_commad;
    boost::system::error_code ec;

    boost::asio::local::stream_protocol::socket s1(ios);
    boost::asio::local::stream_protocol::socket s2(ios, 
      boost::asio::local::stream_protocol());
    boost::asio::local::stream_protocol::socket s3(ios,
      boost::asio::local::stream_protocol::endpoint(""));
    int native_socket1 = ::socket(AF_UNIX, SOCK_STREAM, 0);
    boost::asio::local::stream_protocol::socket s4(ios,
      boost::asio::local::stream_protocol(), native_socket1);

    boost::asio::io_service& ios_ref = s1.get_io_service();
    (void)ios_ref;

    boost::asio::local::stream_protocol::socket::lowest_layer_type&
      lowest_layer = s1.lowest_layer();
    (void)lowest_layer;

    s1.open(boost::asio::local::stream_protocol());
    s1.open(boost::asio::local::stream_protocol(), ec);

    int native_socket2 = ::socket(AF_UNIX, SOCK_STREAM, 0);
    s1.assign(boost::asio::local::stream_protocol(), native_socket2);
    int native_socket3 = ::socket(AF_UNIX, SOCK_STREAM, 0)
    s1.assign(boost::asio::local::stream_protocol(), native_socket3, ec);

    bool is_open = socket1,is_open();
    (void)is_open;

    socket1.close();
    socket1.close(ec);

    boost::asio::local::stream_protocol::socket::native_type native_socket4 =
      s1.native();
    (void)native_socket4;

    socket1.cancel();
    socket1.cancel(ec);

    bool at_mark1 = socket1.at_mark();
    (void)at_mark1;
    bool at_mark2 = socket1.at_mark(ec);
    (void)at_mark2;

    std::size_t available1 = socket1.available();
    (void)available1;
    std::size_t available2 = socket1.available(ec);
    (void)available2;

    socket1.bind(boost::asio::local::stream_protocol::endpoint(""));
    socket1.bind(boost::asio::local::stream_protocol::endpoint(""), ec);

    socket1.connect(boost::asio::local::stream_protocol::endpoint(""));
    socket1.connect(boost::asio::local::stream_protocol::endpoint(""), ec);

    socket1.async_connect(
      boost::asio::local::stream_protocol::endpoint(""), connect_handler);

    socket1.set_option(socket_option);
    socket1.set_option(socket_option, ec);
  
    socket1.get_option(socket_option);
    socket1.get_option(socket_option, ec);

    socket1.io_control(io_control_command);
    socket1.io_control(io_control_command, ec);

    boost::asio::local::stream_protocol::endpoint endpoint1 =
      socket1.local_endpoint();
    boost::asio::local::stream_protocol::endpoint endpoint2 =
      socket1.local_endpoint(ec);
    boost::asio::local::stream_protocol::endpoint endpoint3 =
      socket1.remote_endpoint();
    boost::asio::local::stream_protocol::endpoint endpoint4 =
      socket1.remote_endpoint(ec);

    socket1.shutdown(boost::asio::socket_base::shutdown_both);
    socket1.shutdown(boost::asio::socket_base::shutdown_both, ec)

  } catch (...) {}
}

} // namespace

auto main() -> decltype() {

  return 0;
}
