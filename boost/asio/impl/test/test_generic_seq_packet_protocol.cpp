#include <boost/asio/generic/seq_packet_protocol.hpp>

#include <boost/asio/io_service.hpp>

#include <cstring>

namespace {

void connect_handler(const boost::system::error_code&) {}

void send_handler(const boost::system::error_code&, std::size_t) {}

void receive_handler(const boost::system::error_code&, std::size_t) {}

void test_1() {
  const int af_inet = AF_INET;
  const int sock_seqpacket = SOCK_SEQPACKET;

  try {
    boost::asio::io_service ios;
    char mutable_char_buffer[128] = "";
    const char const_char_buffer[128] = "";
    const boost::asio::socket_base::message_flags in_flags = 0;
    boost::asio::socket_base::message_flags out_flags = 0;
    boost::asio::socket_base::send_buffer_size socket_option;
    boost::asio::socket_base::bytes_readable io_control_command;
    boost::system::error_code ec;

    boost::asio::generic::seq_packet_protocol::socket socket1(ios);
    boost::asio::generic::seq_packet_protocol::socket socket2(ios,
      boost::asio::generic::seq_packet_protocol(af_inet, 0));
    boost::asio::generic::seq_packet_protocol::socket socket3(ios,
      boost::asio::generic::seq_packet_protocol::endpoint());

    boost::asio::generic::seq_packet_protocol::socket::native_handle_type
      native_socket1 = ::socket(af_inet, sock_seqpacket, 0);
    boost::asio::generic::seq_packet_protocol::socket socket4(ios,
      boost::asio::generic::seq_packet_protocol(af_inet, 0), native_socket1);

    boost::asio::generic::seq_packet_protocol::socket socket5(
      std::move(socket4));

    socket1 = boost::asio::generic::seq_packet_protocol::socket(ios);
    socket1 = std::move(socket2);

    boost::asio::io_service& ios_ref = socket1.get_io_service();
    (void)ios_ref;

    boost::asio::generic::seq_packet_protocol::socket::lowest_layer_type&
      lowest_layer = socket1.lowest_layer();
    (void)lowest_layer;

    boost::asio::generic::seq_packet_protocol::socket::native_handle_type
      native_socket2 = ::socket(af_inet, sock_seqpacket, 0);
    socket1.assign(
      boost::asio::generic::seq_packet_protocol(af_inet, 0), native_socket2);
    boost::asio::generic::seq_packet_protocol::socket::native_handle_type
      native_socket3 = ::socket(af_inet, sock_seqpacket, 0);
    socket1.assign(boost::asio::generic::seq_packet_protocol(af_inet, 0),
      native_socket3, ec);

    bool is_open = socket1.is_open();
    (void)is_open;

    socket1.close();
    socket1.close(ec);

    boost::asio::generic::seq_packet_protocol::socket::native_type
      native_socket4 = socket1.native();

    socket1.cancel();
    socket1.cancel(ec);

    bool at_mark1 = socket1.at_mark();
    (void)at_mark1;
    bool at_mark2 = socket1.at_mark(ec);
    (void)at_mark2;

    socket1.bind(boost::asio::generic::seq_packet_protocol::endpoint());
    socket1.bind(boost::asio::generic::seq_packet_protocol::endpoint(), ec);

    socket1.connect(boost::asio::generic::seq_packet_protocol::endpoint());
    socket1.connect(boost::asio::generic::seq_packet_protocol::endpoint(), ec);

    socket1.async_connect(
      boost::asio::generic::seq_packet_protocol::endpoint(), connect_handler);

    socket1.set_option(socket_option);
    socket1.set_option(socket_option, ec);

    socket1.get_option(socket_option);
    socket1.get_option(socket_option, ec);

    socket1.io_control(io_control_command);
    socket1.io_control(io_control_command, ec);

    boost::asio::generic::seq_packet_protocol::endpoint endpoint1 =
      socket1.local_endpoint();
    boost::asio::generic::seq_packet_protocol::endpoint endpoint2 =
      socket1.local_endpoint(ec);

    boost::asio::generic::seq_packet_protocol::endpoint endpoint3 =
      socket1.remote_endpoint();
    boost::asio::generic::seq_packet_protocol::endpoint endpoint4 =
      socket1.remote_endpoint(ec);

    socket1.shutdown(boost::asio::socket_base::shutdown_both);
    socket1.shutdown(boost::asio::socket_base::shutdown_both, ec);

    socket1.send(
      boost::asio::buffer(mutable_char_buffer), in_flags);
    socket1.send(
      boost::asio::buffer(mutable_char_buffer), in_flags, ec);
    socket1.send(
      boost::asio::buffer(const_char_buffer), in_flags);
    socket1.send(
      boost::asio::buffer(const_char_buffer), in_flags, ec);
    socket1.send(
      boost::asio::null_buffers(), in_flags);
    socket1.send(
      boost::asio::null_buffers(), in_flags, ec);

    socket1.async_send(
      boost::asio::buffer(mutable_char_buffer), in_flags, send_handler);
    socket1.async_send(
      boost::asio::buffer(const_char_buffer), in_flags, send_handler);
    socket1.async_send(
      boost::asio::null_buffers(), in_flags, send_handler);

    socket1.receive(
      boost::asio::buffer(mutable_char_buffer), out_flags);
    socket1.receive(
      boost::asio::buffer(mutable_char_buffer), in_flags, out_flags);
    socket1.receive(
      boost::asio::buffer(mutable_char_buffer), in_flags, out_flags, ec);
    socket1.receive(
      boost::asio::null_buffers(), out_flags);
    socket1.receive(
      boost::asio::null_buffers(), in_flags, out_flags);
    socket1.receive(
      boost::asio::null_buffers(), in_flags, out_flags, ec);

    socket1.async_receive(
      boost::asio::buffer(mutable_char_buffer), out_flags, receive_handler);
    socket1.async_receive(
      boost::asio::buffer(mutable_char_buffer),
      in_flags, out_flags, receive_handler);
    socket1.async_receive(
      boost::asio::null_buffers(), out_flags, receive_handler);
    socket1.async_receive(
      boost::asio::null_buffers(), in_flags, out_flags, receive_handler);
  } catch (...) {}
}
} // namespace

auto main() -> decltype(0) {
  test_1();
  return 0;
}
