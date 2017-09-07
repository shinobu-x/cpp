#include <boost/asio/ip/udp.hpp>

#include <boost/asio/io_service.hpp>
#include <boost/bind.hpp>

#include <cstring>
#include <cassert>

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
    s1.bind(boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), 0), ec);
    s1.bind(boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v6(), 0), ec);
    s1.connect(
      boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), 0));
    s1.connect(
      boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v6(), 0));
    s1.connect(
      boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), 0), ec);
    s1.connect(
      boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v6(), 0), ec);
    int l1 = s1.async_connect(
      boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), 0), lazy);
    (void)l1;
    int l2 = s1.async_connect(
      boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v6(), 0), lazy);
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
    boost::asio::ip::udp::endpoint ep1 = s1.local_endpoint();
    boost::asio::ip::udp::endpoint ep2 = s1.local_endpoint(ec);
    boost::asio::ip::udp::endpoint ep3 = s1.remote_endpoint();
    boost::asio::ip::udp::endpoint ep4 = s1.remote_endpoint(ec);
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
    s1.async_send(boost::asio::buffer(mutable_char_buffer), &send_handler);
    s1.async_send(boost::asio::buffer(const_char_buffer), &send_handler);
    s1.async_send(boost::asio::null_buffers(), &send_handler);
    s1.async_send(
      boost::asio::buffer(mutable_char_buffer), in_flags, &send_handler);
    s1.async_send(
      boost::asio::buffer(const_char_buffer), in_flags, &send_handler);
    s1.async_send(
      boost::asio::null_buffers(), in_flags, &send_handler);
    int l3 = s1.async_send(
      boost::asio::buffer(mutable_char_buffer), lazy);
    (void)l3;
    int l4 = s1.async_send(
      boost::asio::buffer(const_char_buffer), lazy);
    (void)l4;
    int l5 = s1.async_send(boost::asio::null_buffers(), lazy);
    (void)l5;
    int l6 = s1.async_send(
      boost::asio::buffer(mutable_char_buffer), in_flags, lazy);
    (void)l6;
    int l7 = s1.async_send(
      boost::asio::buffer(const_char_buffer), in_flags, lazy);
    (void)l7;
    int l8 = s1.async_send(boost::asio::null_buffers(), in_flags, lazy);
    (void)l8;
    s1.send_to(boost::asio::buffer(mutable_char_buffer),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v4(), 0));
    s1.send_to(boost::asio::buffer(mutable_char_buffer),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v4(), 0));
    s1.send_to(boost::asio::buffer(const_char_buffer),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v6(), 0));
    s1.send_to(boost::asio::buffer(const_char_buffer),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v6(), 0));
    s1.send_to(boost::asio::null_buffers(),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v4(), 0));
    s1.send_to(boost::asio::null_buffers(),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v6(), 0));
    s1.send_to(boost::asio::buffer(mutable_char_buffer),
      boost::asio::ip::udp::endpoint(
       boost::asio::ip::udp::v4(), 0), in_flags);
    s1.send_to(boost::asio::buffer(const_char_buffer),
      boost::asio::ip::udp::endpoint(
       boost::asio::ip::udp::v4(), 0), in_flags);
    s1.send_to(boost::asio::buffer(mutable_char_buffer),
      boost::asio::ip::udp::endpoint(
       boost::asio::ip::udp::v6(), 0), in_flags);
    s1.send_to(boost::asio::buffer(const_char_buffer),
      boost::asio::ip::udp::endpoint(
       boost::asio::ip::udp::v6(), 0), in_flags);
    s1.send_to(boost::asio::null_buffers(),
      boost::asio::ip::udp::endpoint(
       boost::asio::ip::udp::v4(), 0), in_flags);
    s1.send_to(boost::asio::null_buffers(),
      boost::asio::ip::udp::endpoint(
       boost::asio::ip::udp::v6(), 0), in_flags);
    s1.send_to(boost::asio::buffer(mutable_char_buffer),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v4(), 0), in_flags, ec);
    s1.send_to(boost::asio::buffer(const_char_buffer),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v4(), 0), in_flags, ec);
    s1.send_to(boost::asio::buffer(mutable_char_buffer),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v6(), 0), in_flags, ec);
    s1.send_to(boost::asio::buffer(const_char_buffer),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v6(), 0), in_flags, ec);
    s1.send_to(boost::asio::null_buffers(),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v4(), 0), in_flags, ec);
    s1.send_to(boost::asio::null_buffers(),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v6(), 0), in_flags, ec);
    s1.async_send_to(boost::asio::buffer(mutable_char_buffer),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v4(), 0), &send_handler);
    s1.async_send_to(boost::asio::buffer(const_char_buffer),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v4(), 0), &send_handler);
    s1.async_send_to(boost::asio::buffer(mutable_char_buffer),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v6(), 0), &send_handler);
    s1.async_send_to(boost::asio::buffer(const_char_buffer),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v6(), 0), &send_handler);
    s1.async_send_to(boost::asio::null_buffers(),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v4(), 0), &send_handler);
    s1.async_send_to(boost::asio::null_buffers(),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v6(), 0), &send_handler);
    s1.async_send_to(boost::asio::buffer(mutable_char_buffer),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v4(), 0), in_flags, &send_handler);
    s1.async_send_to(boost::asio::buffer(const_char_buffer),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v4(), 0), in_flags, &send_handler);
    s1.async_send_to(boost::asio::buffer(mutable_char_buffer),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v6(), 0), in_flags, &send_handler);
    s1.async_send_to(boost::asio::buffer(const_char_buffer),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v6(), 0), in_flags, &send_handler);
    s1.async_send_to(boost::asio::null_buffers(),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v4(), 0), in_flags, &send_handler);
    s1.async_send_to(boost::asio::null_buffers(),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v6(), 0), in_flags, &send_handler);
    int l9 = s1.async_send_to(boost::asio::buffer(mutable_char_buffer),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v4(), 0), lazy);
    (void)l9;
    int l10 = s1.async_send_to(boost::asio::buffer(const_char_buffer),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v4(), 0), lazy);
    (void)l10;
    int l11 = s1.async_send_to(boost::asio::buffer(mutable_char_buffer),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v6(), 0), lazy);
    (void)l11;
    int l12 = s1.async_send_to(boost::asio::buffer(const_char_buffer),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v6(), 0), lazy);
    (void)l12;
    int l13 = s1.async_send_to(boost::asio::null_buffers(),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v4(), 0), lazy);
    (void)l13;
    int l14 = s1.async_send_to(boost::asio::null_buffers(),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v6(), 0), lazy);
    (void)l14;
    int l15 = s1.async_send_to(boost::asio::buffer(mutable_char_buffer),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v4(), 0), in_flags, lazy);
    (void)l15;
    int l16 = s1.async_send_to(boost::asio::buffer(const_char_buffer),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v4(), 0), in_flags, lazy);
    (void)l16;
    int l17 = s1.async_send_to(boost::asio::buffer(mutable_char_buffer),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v6(), 0), in_flags, lazy);
    (void)l17;
    int l18 = s1.async_send_to(boost::asio::buffer(const_char_buffer),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v6(), 0), in_flags, lazy);
    (void)l18;
    int l19 = s1.async_send_to(boost::asio::null_buffers(),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v4(), 0), in_flags, lazy);
    (void)l19;
    int l20 = s1.async_send_to(boost::asio::null_buffers(),
      boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v6(), 0), in_flags, lazy);
    (void)l20;
    s1.receive(boost::asio::buffer(mutable_char_buffer));
    s1.receive(boost::asio::null_buffers());
    s1.receive(boost::asio::buffer(mutable_char_buffer), in_flags);
    s1.receive(boost::asio::null_buffers(), in_flags);
    s1.receive(boost::asio::buffer(mutable_char_buffer), in_flags, ec);
    s1.receive(boost::asio::null_buffers(), in_flags, ec);
    s1.async_receive(
      boost::asio::buffer(mutable_char_buffer), &receive_handler);
    s1.async_receive(
      boost::asio::null_buffers(), &receive_handler);
    s1.async_receive(
      boost::asio::buffer(mutable_char_buffer), in_flags, &receive_handler);
    s1.async_receive(
      boost::asio::null_buffers(), in_flags, &receive_handler);
    int l21 = s1.async_receive(
      boost::asio::buffer(mutable_char_buffer), lazy);
    (void)l21;
    int l22 = s1.async_receive(
      boost::asio::null_buffers(), lazy);
    (void)l22;
    int l23 = s1.async_receive(
      boost::asio::buffer(mutable_char_buffer), in_flags, lazy);
    (void)l23;
    int l24 = s1.async_receive(
      boost::asio::null_buffers(), in_flags, lazy);
    (void)l24;

    boost::asio::ip::udp::endpoint ep;
    s1.receive_from(boost::asio::buffer(mutable_char_buffer), ep);
    s1.receive_from(boost::asio::null_buffers(), ep);
    s1.receive_from(boost::asio::buffer(mutable_char_buffer), ep, in_flags);
    s1.receive_from(boost::asio::null_buffers(), ep, in_flags);
    s1.receive_from(boost::asio::buffer(mutable_char_buffer), ep, in_flags, ec);
    s1.receive_from(boost::asio::null_buffers(), ep, in_flags, ec);
    s1.async_receive_from(
      boost::asio::buffer(mutable_char_buffer), ep, &receive_handler);
    s1.async_receive_from(
      boost::asio::null_buffers(), ep, &receive_handler);
    s1.async_receive_from(
      boost::asio::buffer(mutable_char_buffer), ep, in_flags, &receive_handler);
    s1.async_receive_from(
      boost::asio::null_buffers(), ep, in_flags, &receive_handler);
    int l25 = s1.async_receive_from(
      boost::asio::buffer(mutable_char_buffer), ep, lazy);
    (void)l25;
    int l26 = s1.async_receive_from(boost::asio::null_buffers(), ep, lazy);
    (void)l26;
    int l27 = s1.async_receive_from(
      boost::asio::buffer(mutable_char_buffer), ep, in_flags, lazy);
    (void)l27;
    int l28 = s1.async_receive_from(
      boost::asio::null_buffers(), ep, in_flags, lazy);
    (void)l28;
  } catch (std::exception&) {}
}
} // namespae

namespace {
void handle_send(size_t expected_bytes_sent,
  const boost::system::error_code& ec, size_t bytes_sent) {
  assert(!ec);
  assert(expected_bytes_sent == bytes_sent);
}

void handle_receive(size_t expected_bytes_received,
  const boost::system::error_code& ec, size_t bytes_received) {
  assert(!ec);
  assert(expected_bytes_received == bytes_received);
}

void test_2() {
  boost::asio::io_service ios;
  boost::asio::ip::udp::socket s1(ios,
    boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), 0));
  boost::asio::ip::udp::endpoint target_endpoint = s1.local_endpoint();
  target_endpoint.address(boost::asio::ip::address_v4::loopback());
  boost::asio::ip::udp::socket s2(ios);
  s2.open(boost::asio::ip::udp::v4());
  s2.bind(boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), 0));
  char send_msg[] = "AGCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxz";
  s2.send_to(boost::asio::buffer(send_msg, sizeof(send_msg)), target_endpoint);
  char recv_msg[sizeof(send_msg)];
  boost::asio::ip::udp::endpoint sender_endpoint;
  size_t bytes_received = s1.receive_from(
    boost::asio::buffer(recv_msg, sizeof(recv_msg)), sender_endpoint);
  assert(bytes_received == sizeof(send_msg));
  assert(memcmp(send_msg, recv_msg, sizeof(send_msg)) == 0);
  memset(recv_msg, 0, sizeof(recv_msg));
  target_endpoint = sender_endpoint;
  s1.async_send_to(boost::asio::buffer(send_msg, sizeof(send_msg)),
    target_endpoint, boost::bind(handle_send, sizeof(send_msg), _1, _2));
  s2.async_receive_from(boost::asio::buffer(recv_msg, sizeof(recv_msg)),
    sender_endpoint, boost::bind(handle_receive, sizeof(recv_msg), _1, _2));
  ios.run();
  assert(memcmp(send_msg, recv_msg, sizeof(send_msg)) == 0);
}
} // namespace

namespace {
void resolve_handler(const boost::system::error_code&,
  boost::asio::ip::udp::resolver::iterator) {}

void test_3() {
  try {
    boost::asio::io_service ios;
    lazy_handler lazy;
    boost::system::error_code ec;
    boost::asio::ip::udp::resolver::query q(
      boost::asio::ip::udp::v4(), "localhost", "0");
    boost::asio::ip::udp::endpoint ep(
      boost::asio::ip::address_v4::loopback(), 0);
    boost::asio::ip::udp::resolver r(ios);
    boost::asio::io_service& ios_ref = r.get_io_service();
    (void)ios_ref;
    r.cancel();
    boost::asio::ip::udp::resolver::iterator it1 = r.resolve(q);
    (void)it1;
    boost::asio::ip::udp::resolver::iterator it2 = r.resolve(q, ec);
    (void)it2;
    boost::asio::ip::udp::resolver::iterator it3 = r.resolve(ep);
    (void)it3;
    boost::asio::ip::udp::resolver::iterator it4 = r.resolve(ep, ec);
    (void)it4;
    r.async_resolve(q, &resolve_handler);
    int l1 = r.async_resolve(q, lazy);
    (void)l1;
    r.async_resolve(ep, &resolve_handler);
    int l2 = r.async_resolve(ep, lazy);
    (void)l2;
  } catch (std::exception&) {}
}
} // namespace
auto main() -> decltype(0) {
  test_1(); test_2(); test_3();
  return 0;
}
