#include <boost/asio/generic/stream_protocol.hpp>

#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/tcp.hpp>

#include <cstring>

namespace {

void connect_handler(const boost::system::error_code&) {}

void send_handler(const boost::system::error_code&, std::size_t) {}

void receive_handler(const boost::system::error_code&, std::size_t) {}

void write_some_handler(const boost::system::error_code&, std::size_t) {}

void read_some_handler(const boost::system::error_code&, std::size_t) {}

void test_1() {
  const int af_inet = AF_INET;
  const int ipproto_tcp = IPPROTO_TCP;
  const int sock_stream = SOCK_STREAM;

  try {
    boost::asio::io_service ios;
    char mutable_char_buffer[128] = "";
    const char const_char_buffer[128] = "";
    boost::asio::socket_base::message_flags in_flags = 0;
    boost::asio::socket_base::keep_alive socket_option;
    boost::asio::socket_base::bytes_readable io_control_command;
    boost::system::error_code ec;

    boost::asio::generic::stream_protocol::socket socket1(ios);
    boost::asio::generic::stream_protocol::socket socket2(ios,
      boost::asio::generic::stream_protocol(af_inet, ipproto_tcp));
    boost::asio::generic::stream_protocol::socket socket3(ios,
      boost::asio::generic::stream_protocol::endpoint());

    boost::asio::generic::stream_protocol::socket::native_handle_type
      native_socket1 = ::socket(af_inet, sock_stream, 0);
    boost::asio::generic::stream_protocol::socket socket4(ios,
      boost::asio::generic::stream_protocol(
        af_inet, ipproto_tcp), native_socket1);

    socket1 = boost::asio::generic::stream_protocol::socket(ios);
    socket1 = std::move(socket2);
    socket1 = boost::asio::ip::tcp::socket(ios);

    boost::asio::io_service& ios_ref = socket1.get_io_service();
    (void)ios_ref;

    boost::asio::generic::stream_protocol::socket::lowest_layer_type&
      lowest_layer = socket1.lowest_layer();

    socket1.open(boost::asio::generic::stream_protocol(
      af_inet, ipproto_tcp));
    socket1.open(boost::asio::generic::stream_protocol(
      af_inet, ipproto_tcp), ec);

    boost::asio::generic::stream_protocol::socket::native_handle_type
      native_socket2 = ::socket(af_inet, sock_stream, 0);
    socket1.assign(
      boost::asio::generic::stream_protocol(
        af_inet, ipproto_tcp), native_socket2);

    boost::asio::generic::stream_protocol::socket::native_handle_type
      native_socket3 = ::socket(af_inet, sock_stream, 0);

    bool is_open = socket1.is_open();
    (void)is_open;

    socket1.close();
    socket1.close(ec);

    bool at_mark1 = socket1.at_mark();
    (void)at_mark1;
    bool at_mark2 = socket1.at_mark(ec);
    (void)at_mark2;

    std::size_t available1 = socket1.available();
    (void)available1;
    std::size_t available2 = socket1.available(ec);
    (void)available2;

    socket1.bind(boost::asio::generic::stream_protocol::endpoint());
    socket1.bind(boost::asio::generic::stream_protocol::endpoint(), ec);

    boost::asio::generic::stream_protocol::endpoint endpoint1 =
      socket1.local_endpoint();
    boost::asio::generic::stream_protocol::endpoint endpoint2 =
      socket1.local_endpoint(ec);
    boost::asio::generic::stream_protocol::endpoint endpoint3 =
      socket1.remote_endpoint();
    boost::asio::generic::stream_protocol::endpoint endpoint4 =
      socket1.remote_endpoint();

    socket1.shutdown(boost::asio::socket_base::shutdown_both);
    socket1.shutdown(boost::asio::socket_base::shutdown_both, ec);

    socket1.send(boost::asio::buffer(mutable_char_buffer));
    socket1.send(boost::asio::buffer(mutable_char_buffer), in_flags);
    socket1.send(boost::asio::buffer(mutable_char_buffer), in_flags, ec);
    socket1.send(boost::asio::buffer(const_char_buffer));
    socket1.send(boost::asio::buffer(const_char_buffer), in_flags);
    socket1.send(boost::asio::buffer(const_char_buffer), in_flags, ec);
    socket1.send(boost::asio::null_buffers());
    socket1.send(boost::asio::null_buffers(), in_flags);
    socket1.send(boost::asio::null_buffers(), in_flags, ec);

    socket1.async_send(
      boost::asio::buffer(mutable_char_buffer), send_handler);
    socket1.async_send(
      boost::asio::buffer(mutable_char_buffer), in_flags, send_handler);
    socket1.async_send(
      boost::asio::buffer(const_char_buffer), send_handler);
    socket1.async_send(
      boost::asio::buffer(const_char_buffer), in_flags, send_handler);
    socket1.async_send(
      boost::asio::null_buffers(), send_handler);
    socket1.async_send(
      boost::asio::null_buffers(), in_flags, send_handler);

    socket1.receive(
      boost::asio::buffer(mutable_char_buffer));
    socket1.receive(
      boost::asio::buffer(mutable_char_buffer), in_flags);
    socket1.receive(
      boost::asio::buffer(mutable_char_buffer), in_flags, ec);
    socket1.receive(
      boost::asio::null_buffers());
    socket1.receive(
      boost::asio::null_buffers(), in_flags);
    socket1.receive(
      boost::asio::null_buffers(), in_flags, ec);

    socket1.async_receive(
      boost::asio::buffer(mutable_char_buffer), receive_handler);
    socket1.async_receive(
      boost::asio::buffer(mutable_char_buffer), in_flags, receive_handler);
    socket1.async_receive(
      boost::asio::null_buffers(), receive_handler);
    socket1.async_receive(
      boost::asio::null_buffers(), in_flags, receive_handler);

    socket1.write_some(
      boost::asio::buffer(mutable_char_buffer));
    socket1.write_some(
      boost::asio::buffer(mutable_char_buffer), ec);
    socket1.write_some(
      boost::asio::buffer(const_char_buffer));
    socket1.write_some(
      boost::asio::buffer(const_char_buffer), ec);
    socket1.write_some(
      boost::asio::null_buffers());
    socket1.write_some(
      boost::asio::null_buffers(), ec);

    socket1.async_write_some(
      boost::asio::buffer(mutable_char_buffer), write_some_handler);
    socket1.async_write_some(
      boost::asio::buffer(const_char_buffer), write_some_handler);
    socket1.async_write_some(
      boost::asio::null_buffers(), write_some_handler);

    socket1.read_some(
      boost::asio::buffer(mutable_char_buffer));
    socket1.read_some(
      boost::asio::buffer(mutable_char_buffer), ec);
    socket1.read_some(
      boost::asio::null_buffers());
    socket1.read_some(
      boost::asio::null_buffers(), ec);

    socket1.async_read_some(
      boost::asio::buffer(mutable_char_buffer), read_some_handler);
    socket1.async_read_some(
      boost::asio::null_buffers(), read_some_handler);
  } catch (...) {}
}
} // namespace

auto main() -> decltype(0) {
  test_1();
  return 0;
}
