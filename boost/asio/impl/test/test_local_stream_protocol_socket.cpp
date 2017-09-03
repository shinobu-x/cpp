#include <boost/asio/local/stream_protocol.hpp>
#include <boost/asio/io_service.hpp>

#include <cstring>

void connect_handler(const boost::system::error_code&) {}
void send_handler(const boost::system::error_code&, std::size_t) {}
void receive_handler(const boost::system::error_code&, std::size_t) {}
void read_some_handler(const boost::system::error_code&, std::size_t) {}
void write_some_handler(const boost::system::error_code&, std::size_t) {}

void test_1() {
  try {
    boost::asio::io_service ios;
    char mutable_char_buffer[128] = "";
    const char const_char_buffer[128] = "";
    boost::asio::socket_base::message_flags in_flags = 0;
    boost::asio::socket_base::keep_alive socket_option;
    boost::asio::socket_base::bytes_readable io_control_command;
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
    int native_socket3 = ::socket(AF_UNIX, SOCK_STREAM, 0);
    s1.assign(boost::asio::local::stream_protocol(), native_socket3, ec);

    bool is_open = s1.is_open();
    (void)is_open;

    s1.close();
    s1.close(ec);

    boost::asio::local::stream_protocol::socket::native_type native_socket4 =
      s1.native();
    (void)native_socket4;

    s1.cancel();
    s1.cancel(ec);

    bool at_mark1 = s1.at_mark();
    (void)at_mark1;
    bool at_mark2 = s1.at_mark(ec);
    (void)at_mark2;

    s1.bind(boost::asio::local::stream_protocol::endpoint(""));
    s1.bind(boost::asio::local::stream_protocol::endpoint(""), ec);

    s1.connect(boost::asio::local::stream_protocol::endpoint(""));
    s1.connect(boost::asio::local::stream_protocol::endpoint(""), ec);

    s1.async_connect(
      boost::asio::local::stream_protocol::endpoint(""), connect_handler);

    s1.set_option(socket_option);
    s1.set_option(socket_option, ec);

    s1.get_option(socket_option);
    s1.get_option(socket_option, ec);

    s1.io_control(io_control_command);
    s1.io_control(io_control_command, ec);

    boost::asio::local::stream_protocol::endpoint ep1 =
      s1.local_endpoint();
    boost::asio::local::stream_protocol::endpoint ep2 =
      s2.local_endpoint(ec);

    boost::asio::local::stream_protocol::endpoint ep3 =
      s1.remote_endpoint();
    boost::asio::local::stream_protocol::endpoint ep4 =
      s1.remote_endpoint();

    s1.shutdown(boost::asio::socket_base::shutdown_both);
    s1.shutdown(boost::asio::socket_base::shutdown_both, ec);

    s1.send(boost::asio::buffer(mutable_char_buffer));
    s1.send(boost::asio::buffer(const_char_buffer));
    s1.send(boost::asio::null_buffers());
    s1.send(boost::asio::buffer(mutable_char_buffer), in_flags);
    s1.send(boost::asio::buffer(const_char_buffer), in_flags);
    s1.send(boost::asio::null_buffers(), in_flags);
    s1.send(boost::asio::buffer(mutable_char_buffer), in_flags, ec);
    s1.send(boost::asio::buffer(const_char_buffer), in_flags, ec);
    s1.send(boost::asio::null_buffers(), in_flags, ec);

    s1.async_send(boost::asio::buffer(mutable_char_buffer), send_handler);
    s1.async_send(boost::asio::buffer(const_char_buffer), send_handler);
    s1.async_send(boost::asio::null_buffers(), send_handler);
    s1.async_send(boost::asio::buffer(mutable_char_buffer),
      in_flags, send_handler);
    s1.async_send(boost::asio::buffer(const_char_buffer),
      in_flags, send_handler);
    s1.async_send(boost::asio::null_buffers(), in_flags, send_handler);

    s1.receive(boost::asio::buffer(mutable_char_buffer));
    s1.receive(boost::asio::null_buffers());
    s1.receive(boost::asio::buffer(mutable_char_buffer), in_flags);
    s1.receive(boost::asio::null_buffers(), in_flags);
    s1.receive(boost::asio::buffer(mutable_char_buffer), in_flags, ec);
    s1.receive(boost::asio::null_buffers(), in_flags, ec);

    s1.async_receive(boost::asio::buffer(mutable_char_buffer),
      receive_handler);
    s1.async_receive(boost::asio::null_buffers(), receive_handler);
    s1.async_receive(boost::asio::buffer(mutable_char_buffer), in_flags,
      receive_handler);
    s1.async_receive(boost::asio::null_buffers(), in_flags, receive_handler);

    s1.write_some(boost::asio::buffer(mutable_char_buffer));
    s1.write_some(boost::asio::buffer(const_char_buffer));
    s1.write_some(boost::asio::null_buffers());
    s1.write_some(boost::asio::buffer(mutable_char_buffer), ec);
    s1.write_some(boost::asio::buffer(const_char_buffer), ec);
    s1.write_some(boost::asio::null_buffers(), ec);

    s1.async_write_some(boost::asio::buffer(mutable_char_buffer),
      write_some_handler);
    s1.async_write_some(boost::asio::buffer(const_char_buffer),
      write_some_handler);
    s1.async_write_some(boost::asio::null_buffers(), write_some_handler);

    s1.read_some(boost::asio::buffer(mutable_char_buffer));
    s1.read_some(boost::asio::buffer(mutable_char_buffer), ec);
    s1.read_some(boost::asio::null_buffers(), ec);

    s1.async_read_some(boost::asio::buffer(mutable_char_buffer),
      read_some_handler);
    s1.async_read_some(boost::asio::null_buffers(), read_some_handler);
  } catch (std::exception&) {}
}

auto main() -> decltype(0) {
  return 0;
}
