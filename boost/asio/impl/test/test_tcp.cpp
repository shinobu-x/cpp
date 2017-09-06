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
    int l1 = s1.async_connect(boost::asio::ip::tcp::endpoint(
      boost::asio::ip::tcp::v4(), 0), lazy);
    (void)l1;
    int l2 = s1.async_connect(boost::asio::ip::tcp::endpoint(
      boost::asio::ip::tcp::v6(), 0), lazy);
    (void)l2;
    s1.set_option(settable_socket_option1);
    s1.set_option(settable_socket_option1, ec);
    s1.set_option(settable_socket_option2);
    s1.set_option(settable_socket_option2, ec);
    s1.set_option(settable_socket_option3);
    s1.set_option(settable_socket_option3, ec);
    s1.get_option(gettable_socket_option1);
    s1.get_option(gettable_socket_option1, ec);
    s1.get_option(gettable_socket_option2);
    s1.get_option(gettable_socket_option2, ec);
    s1.get_option(gettable_socket_option3);
    s1.get_option(gettable_socket_option3, ec);
    s1.io_control(io_control_command);
    s1.io_control(io_control_command, ec);
    bool non_blocking1 = s1.non_blocking();
    (void)non_blocking1;
    s1.non_blocking(true);
    s1.non_blocking(false, ec);
    bool non_blocking2 = s1.native_non_blocking();
    (void)non_blocking2;
    s1.native_non_blocking(true);
    s1.native_non_blocking(false, ec);
    boost::asio::ip::tcp::endpoint ep1 = s1.local_endpoint();
    boost::asio::ip::tcp::endpoint ep2 = s1.local_endpoint(ec);
    boost::asio::ip::tcp::endpoint ep3 = s1.remote_endpoint();
    boost::asio::ip::tcp::endpoint ep4 = s1.remote_endpoint(ec);
    s1.shutdown(boost::asio::socket_base::shutdown_both);
    s1.shutdown(boost::asio::socket_base::shutdown_both, ec);
    s1.send(boost::asio::buffer(mutable_char_buffer));
    s1.send(boost::asio::buffer(const_char_buffer));
    s1.send(mutable_buffers);
    s1.send(const_buffers);
    s1.send(boost::asio::null_buffers());
    s1.send(boost::asio::buffer(mutable_char_buffer), in_flags);
    s1.send(boost::asio::buffer(const_char_buffer), in_flags);
    s1.send(mutable_buffers, in_flags);
    s1.send(const_buffers, in_flags);
    s1.send(boost::asio::null_buffers(), in_flags);
    s1.send(boost::asio::buffer(mutable_char_buffer), in_flags, ec);
    s1.send(boost::asio::buffer(const_char_buffer), in_flags, ec);
    s1.send(mutable_buffers, in_flags, ec);
    s1.send(const_buffers, in_flags, ec);
    s1.send(boost::asio::null_buffers(), in_flags, ec);
    s1.async_send(boost::asio::buffer(mutable_char_buffer), &send_handler);
    s1.async_send(boost::asio::buffer(const_char_buffer), &send_handler);
    s1.async_send(mutable_buffers, &send_handler);
    s1.async_send(const_buffers, &send_handler);
    s1.async_send(boost::asio::null_buffers(), &send_handler);
    s1.async_send(boost::asio::null_buffers(), &send_handler);
    s1.async_send(
      boost::asio::buffer(mutable_char_buffer), in_flags, &send_handler);
    s1.async_send(
      boost::asio::buffer(const_char_buffer), in_flags, &send_handler);
    s1.async_send(mutable_buffers, in_flags, &send_handler);
    s1.async_send(const_buffers, in_flags, &send_handler);
    s1.async_send(boost::asio::null_buffers(), in_flags, &send_handler);
    int l3 = s1.async_send(boost::asio::buffer(mutable_char_buffer), lazy);
    (void)l3;
    int l4 = s1.async_send(boost::asio::buffer(const_char_buffer), lazy);
    (void)l4;
    int l5 = s1.async_send(mutable_buffers, lazy);
    (void)l5;
    int l6 = s1.async_send(const_buffers, lazy);
    (void)l6;
    int l7 = s1.async_send(boost::asio::null_buffers(), lazy);
    (void)l7;
    int l8 = s1.async_send(
      boost::asio::buffer(mutable_char_buffer), in_flags, lazy);
    (void)l8;
    int l9 = s1.async_send(
      boost::asio::buffer(const_char_buffer), in_flags, lazy);
    (void)l9;
    int l10 = s1.async_send(mutable_buffers, in_flags, lazy);
    (void)l10;
    int l11 = s1.async_send(const_buffers, in_flags, lazy);
    (void)l11;
    int l12 = s1.async_send(boost::asio::null_buffers(), in_flags, lazy);
    (void)l12;
    s1.receive(boost::asio::buffer(mutable_char_buffer));
    s1.receive(mutable_buffers);
    s1.receive(boost::asio::null_buffers());
    s1.receive(boost::asio::buffer(mutable_char_buffer), in_flags);
    s1.receive(mutable_buffers, in_flags);
    s1.receive(boost::asio::null_buffers(), in_flags);
    s1.receive(boost::asio::buffer(mutable_char_buffer), in_flags, ec);
    s1.receive(mutable_buffers, in_flags, ec);
    s1.receive(boost::asio::null_buffers(), in_flags, ec);
    s1.async_receive(
      boost::asio::buffer(mutable_char_buffer), &receive_handler);
    s1.async_receive(mutable_buffers, &receive_handler);
    s1.async_receive(boost::asio::null_buffers(), &receive_handler);
    s1.async_receive(
      boost::asio::buffer(mutable_char_buffer), in_flags, &receive_handler);
    s1.async_receive(mutable_buffers, in_flags, &receive_handler);
    s1.async_receive(boost::asio::null_buffers(), in_flags, &receive_handler);
    int l13 = s1.async_receive(boost::asio::buffer(mutable_char_buffer), lazy);
    (void)l13;
    int l14 = s1.async_receive(mutable_buffers, lazy);
    (void)l14;
    int l15 = s1.async_receive(boost::asio::null_buffers(), lazy);
    (void)l15;
    int l16 = s1.async_receive(
      boost::asio::buffer(mutable_char_buffer), in_flags, lazy);
    (void)l16;
    int l17 = s1.async_receive(mutable_buffers, in_flags, lazy);
    (void)l17;
    int l18 = s1.async_receive(boost::asio::null_buffers(), in_flags, lazy);
    (void)l18;
    s1.write_some(boost::asio::buffer(mutable_char_buffer));
    s1.write_some(boost::asio::buffer(const_char_buffer));
    s1.write_some(mutable_buffers);
    s1.write_some(const_buffers);
    s1.write_some(boost::asio::null_buffers());
    s1.write_some(boost::asio::buffer(mutable_char_buffer), ec);
    s1.write_some(boost::asio::buffer(const_char_buffer), ec);
    s1.write_some(mutable_buffers, ec);
    s1.write_some(const_buffers, ec);
    s1.write_some(boost::asio::null_buffers(), ec);
    s1.async_write_some(
      boost::asio::buffer(mutable_char_buffer), &write_some_handler);
    s1.async_write_some(
      boost::asio::buffer(const_char_buffer), &write_some_handler);
    s1.async_write_some(mutable_buffers, &write_some_handler);
    s1.async_write_some(const_buffers, &write_some_handler);
    s1.async_write_some(boost::asio::null_buffers(), &write_some_handler);
    int l19 = s1.async_write_some(
      boost::asio::buffer(mutable_char_buffer), lazy);
    (void)l19;
    int l20 = s1.async_write_some(
      boost::asio::buffer(const_char_buffer), lazy);
    (void)l20;
    int l21 = s1.async_write_some(mutable_buffers, lazy);
    (void)l21;
    int l22 = s1.async_write_some(const_buffers, lazy);
    (void)l22;
    int l23 = s1.async_write_some(boost::asio::null_buffers(), lazy);
    (void)l23;
    s1.read_some(boost::asio::buffer(mutable_char_buffer));
    s1.read_some(mutable_buffers);
    s1.read_some(boost::asio::null_buffers());
    s1.read_some(boost::asio::buffer(mutable_char_buffer), ec);
    s1.read_some(mutable_buffers, ec);
    s1.read_some(boost::asio::null_buffers(), ec);
    s1.async_read_some(
      boost::asio::buffer(mutable_char_buffer), &read_some_handler);
    s1.async_read_some(mutable_buffers, &read_some_handler);
    s1.async_read_some(boost::asio::null_buffers(), &read_some_handler);
    int l24 = s1.async_read_some(
      boost::asio::buffer(mutable_char_buffer), lazy);
    (void)l24;
    int l25 = s1.async_read_some(mutable_buffers, lazy);
    (void)l25;
    int l26 = s1.async_read_some(boost::asio::null_buffers(), lazy);
    (void)l26;
  } catch (std::exception&) {}
}

namespace {
static const char write_data[] =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

void handle_read_noop(const boost::system::error_code& ec,
  size_t bytes_transferred, bool* called) {
  *called = true;
  assert(!ec);
  assert(bytes_transferred == 0);
}

void handle_write_noop(const boost::system::error_code& ec,
  size_t bytes_transferred, bool* called) {
  *called = true;
  assert(!ec);
  assert(bytes_transferred == 0);
}

void handle_read(const boost::system::error_code& ec,
  size_t bytes_transferred, bool* called) {
  *called = true;
  assert(!ec);
  assert(bytes_transferred == sizeof(write_data));
}

void handle_write(const boost::system::error_code& ec,
  size_t bytes_transferred, bool* called) {
  *called = true;
  assert(!ec);
  assert(bytes_transferred == sizeof(write_data));
}

void handle_read_cancel(const boost::system::error_code& ec,
  size_t bytes_transferred, bool* called) {
  *called = true;
  assert(ec == boost::asio::error::operation_aborted);
  assert(bytes_transferred == 0);
}

void handle_read_eof(const boost::system::error_code& ec,
  size_t bytes_transferred, bool* called) {
  *called = true;
  assert(ec == boost::asio::error::eof);
  assert(bytes_transferred == 0);
}
} // namespace

void test_4() {
  boost::asio::io_service ios;
  boost::asio::ip::tcp::acceptor ap(ios,
    boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), 0));
  boost::asio::ip::tcp::endpoint server_endpoint = ap.local_endpoint();
  server_endpoint.address(boost::asio::ip::address_v4::loopback());
  boost::asio::ip::tcp::socket client_socket(ios);
  boost::asio::ip::tcp::socket server_socket(ios);
  client_socket.connect(server_endpoint);
  ap.accept(server_socket);
  bool read_noop_completed = false;
  client_socket.async_read_some(boost::asio::mutable_buffers_1(0, 0),
    boost::bind(handle_read_noop, _1, _2, &read_noop_completed));
  ios.run();
  assert(read_noop_completed);
  bool write_noop_completed = false;
  client_socket.async_write_some(boost::asio::mutable_buffers_1(0, 0),
    boost::bind(handle_write_noop, _1, _2, &write_noop_completed));
  ios.reset();
  ios.run();
  assert(write_noop_completed);
  char read_buffer[sizeof(write_data)];
  bool read_completed = false;
  boost::asio::async_read(client_socket,
    boost::asio::buffer(read_buffer),
    boost::bind(handle_read, _1, _2, &read_completed));
  bool write_completed = false;
  boost::asio::async_write(server_socket,
    boost::asio::buffer(write_data),
    boost::bind(handle_write, _1, _2, &write_completed));
  ios.reset();
  ios.run();
  assert(read_completed);
  assert(write_completed);
  assert(memcmp(read_buffer, write_data, sizeof(write_data)) == 0);
  bool read_cancel_completed = false;
  boost::asio::async_read(server_socket,
    boost::asio::buffer(read_buffer),
    boost::bind(handle_read_cancel, _1, _2, &read_cancel_completed));
  ios.reset();
  ios.poll();
  assert(!read_cancel_completed);
  server_socket.cancel();
  ios.reset();
  ios.poll();
  assert(read_cancel_completed);
  bool read_eof_completed = false;
  boost::asio::async_read(client_socket,
    boost::asio::buffer(read_buffer),
    boost::bind(handle_read_eof, _1, _2, &read_eof_completed));
  server_socket.close();
  ios.reset();
  ios.run();
  assert(read_eof_completed);
}

namespace {
void accept_handler(const boost::system::error_code&) {}
} // namespace

void test_5() {
  try {
    boost::asio::io_service ios;
    boost::asio::ip::tcp::socket peer_socket(ios);
    boost::asio::ip::tcp::endpoint peer_endpoint;
    settable_socket_option<void> settable_socket_option1;
    settable_socket_option<int> settable_socket_option2;
    settable_socket_option<double> settable_socket_option3;
    gettable_socket_option<void> gettable_socket_option1;
    gettable_socket_option<int> gettable_socket_option2;
    gettable_socket_option<double> gettable_socket_option3;
    io_control_command io_control_command;
    lazy_handler lazy;
    boost::system::error_code ec;
    boost::asio::ip::tcp::acceptor ap1(ios);
    boost::asio::ip::tcp::acceptor ap2(ios, boost::asio::ip::tcp::v4());
    boost::asio::ip::tcp::acceptor ap3(ios, boost::asio::ip::tcp::v6());
    boost::asio::ip::tcp::acceptor ap4(ios,
      boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), 0));
    boost::asio::ip::tcp::acceptor ap5(ios,
      boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v6(), 0));
#if !defined(BOOST_ASIO_WINDOWS_RUNTIME)
    boost::asio::ip::tcp::acceptor::native_handle_type native_acceptor1 =
      ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    boost::asio::ip::tcp::acceptor ap6(ios,
      boost::asio::ip::tcp::v4(), native_acceptor1);
#endif
#if defined(BOOST_ASIO_HAS_MOVE)
    boost::asio::ip::tcp::acceptor ap7(std::move(ap5));
#endif
#if defined(BOOST_ASIO_HAS_MOVE)
    ap1 = boost::asio::ip::tcp::acceptor(ios);
    ap1 = std::move(ap2);
#endif
    boost::asio::io_service& ios_ref = ap1.get_io_service();
    (void)ios_ref;
    ap1.open(boost::asio::ip::tcp::v4());
    ap1.open(boost::asio::ip::tcp::v6());
    ap1.open(boost::asio::ip::tcp::v4(), ec);
    ap1.open(boost::asio::ip::tcp::v6(), ec);
#if !defined(BOOST_ASIO_WINDOWS_RUNTIME)
    boost::asio::ip::tcp::acceptor::native_handle_type native_acceptor2 =
      ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    ap1.assign(boost::asio::ip::tcp::v4(), native_acceptor2);
    boost::asio::ip::tcp::acceptor::native_handle_type native_acceptor3 =
      ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    ap1.assign(boost::asio::ip::tcp::v4(), native_acceptor3, ec);
#endif
    bool is_open = ap1.is_open();
    (void)is_open;
    ap1.close();
    ap1.close(ec);
    boost::asio::ip::tcp::acceptor::native_type native_acceptor4 = ap1.native();
    (void)native_acceptor4;
    boost::asio::ip::tcp::acceptor::native_handle_type native_acceptor5 =
      ap1.native_handle();
    (void)native_acceptor5;
    ap1.cancel();
    ap1.cancel(ec);
    ap1.bind(boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), 0));
    ap1.bind(boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v6(), 0));
    ap1.bind(
      boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), 0), ec);
    ap1.bind(
      boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v6(), 0), ec);
    ap1.set_option(settable_socket_option1);
    ap1.set_option(settable_socket_option1, ec);
    ap1.set_option(settable_socket_option2);
    ap1.set_option(settable_socket_option2, ec);
    ap1.set_option(settable_socket_option3);
    ap1.set_option(settable_socket_option3, ec);
    ap1.get_option(gettable_socket_option1);
    ap1.get_option(gettable_socket_option1, ec);
    ap1.get_option(gettable_socket_option2);
    ap1.get_option(gettable_socket_option2, ec);
    ap1.get_option(gettable_socket_option3);
    ap1.get_option(gettable_socket_option3, ec);
    ap1.io_control(io_control_command);
    ap1.io_control(io_control_command, ec);
    bool non_blocking1 = ap1.non_blocking();
    (void)non_blocking1;
    ap1.non_blocking(true);
    ap1.non_blocking(false, ec);
    bool non_blocking2 = ap1.native_non_blocking();
    (void)non_blocking2;
    ap1.native_non_blocking(true);
    ap1.native_non_blocking(false, ec);
    boost::asio::ip::tcp::endpoint ep1 = ap1.local_endpoint();
    boost::asio::ip::tcp::endpoint ep2 = ap1.local_endpoint(ec);
    ap1.accept(peer_socket);
    ap1.accept(peer_socket, ec);
    ap1.accept(peer_socket, peer_endpoint);
    ap1.accept(peer_socket, peer_endpoint, ec);
    ap1.async_accept(peer_socket, &accept_handler);
    ap1.async_accept(peer_socket, peer_endpoint, &accept_handler);
    int l1 = ap1.async_accept(peer_socket, lazy);
    (void)l1;
    int l2 = ap1.async_accept(peer_socket, peer_endpoint, lazy);
    (void)l2;

  } catch (std::exception&) {}
}

namespace {
void handle_accept(const boost::system::error_code& ec) {
  assert(!ec);
}
void handle_connect(const boost::system::error_code& ec) {
  assert(!ec);
}

void test_6() {
  boost::asio::io_service ios;
  boost::asio::ip::tcp::acceptor ap(ios,
    boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), 0));
  boost::asio::ip::tcp::endpoint server_endpoint = ap.local_endpoint();
  server_endpoint.address(boost::asio::ip::address_v4::loopback());
  boost::asio::ip::tcp::socket client_socket(ios);
  boost::asio::ip::tcp::socket server_socket(ios);
  client_socket.connect(server_endpoint);
  ap.accept(server_socket);
  client_socket.close();
  server_socket.close();
  client_socket.connect(server_endpoint);
  boost::asio::ip::tcp::endpoint client_endpoint;
  ap.accept(server_socket, client_endpoint);
  boost::asio::ip::tcp::acceptor::non_blocking_io command(false);
  ap.io_control(command);
  boost::asio::ip::tcp::endpoint client_local_endpoint =
    client_socket.local_endpoint();
  assert(client_local_endpoint.port() == client_endpoint.port());
  boost::asio::ip::tcp::endpoint server_remote_endpoint =
    server_socket.remote_endpoint();
  assert(server_remote_endpoint.port() == client_endpoint.port());
  client_socket.close();
  server_socket.close();
  ap.async_accept(server_socket, client_endpoint, &handle_accept);
  client_socket.async_connect(server_endpoint, &handle_connect);
  ios.reset();
  ios.run();
  client_local_endpoint = client_socket.local_endpoint();
  assert(client_local_endpoint.port() == client_endpoint.port());
  server_remote_endpoint = server_socket.remote_endpoint();
  assert(server_remote_endpoint.port() == client_endpoint.port());
}
} // namespace
auto main() -> decltype(0) {
test_1(); test_2(); test_3(); test_4(); test_5(); test_6();
return 0;
}
