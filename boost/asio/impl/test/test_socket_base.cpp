#include <boost/asio/socket_base.hpp>

#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ip/udp.hpp>

#include <cassert>

namespace compile {
void test_socket_base() {

  try {
    boost::asio::io_service ios;
    boost::asio::ip::tcp::socket sk(ios);
    char buf[128];

    sk.shutdown(boost::asio::socket_base::shutdown_receive);
    sk.shutdown(boost::asio::socket_base::shutdown_send);
    sk.shutdown(boost::asio::socket_base::shutdown_both);

    sk.receive(
      boost::asio::buffer(buf), boost::asio::socket_base::message_peek);
    sk.receive(
      boost::asio::buffer(buf), boost::asio::socket_base::message_out_of_band);
    sk.send(
      boost::asio::buffer(buf), boost::asio::socket_base::message_do_not_route);

    boost::asio::socket_base::broadcast broadcast1(true);
    sk.set_option(broadcast1);
    boost::asio::socket_base::broadcast broadcast2;
    sk.get_option(broadcast2);
    broadcast1 = true;
    (void)static_cast<bool>(broadcast1);
    (void)static_cast<bool>(!broadcast1);
    (void)static_cast<bool>(broadcast1.value());

    boost::asio::socket_base::debug debug1(true);
    sk.set_option(debug1);
    boost::asio::socket_base::debug debug2;
    sk.get_option(debug2);
    debug1 = true;
    (void)static_cast<bool>(debug1);
    (void)static_cast<bool>(!debug1);
    (void)static_cast<bool>(debug1.value());

    boost::asio::socket_base::do_not_route do_not_route1(true);
    sk.set_option(do_not_route1);
    boost::asio::socket_base::do_not_route do_not_route2;
    sk.get_option(do_not_route2);
    do_not_route1 = true;
    (void)static_cast<bool>(do_not_route1);
    (void)static_cast<bool>(!do_not_route1);
    (void)static_cast<bool>(do_not_route1.value());

    boost::asio::socket_base::keep_alive keep_alive1(true);
    sk.set_option(keep_alive1);
    boost::asio::socket_base::keep_alive keep_alive2;
    sk.get_option(keep_alive2);
    keep_alive1 = true;
    (void)static_cast<bool>(keep_alive1);
    (void)static_cast<bool>(!keep_alive1);
    (void)static_cast<bool>(keep_alive1.value());

    boost::asio::socket_base::send_buffer_size send_buffer_size1(128);
    sk.set_option(send_buffer_size1);
    boost::asio::socket_base::send_buffer_size send_buffer_size2;
    sk.get_option(send_buffer_size2);
    send_buffer_size1 = 1;
    (void)static_cast<int>(send_buffer_size1.value());

    boost::asio::socket_base::send_low_watermark send_low_watermark1(64);
    sk.set_option(send_low_watermark1);
    boost::asio::socket_base::send_low_watermark send_low_watermark2;
    sk.get_option(send_low_watermark2);
    send_low_watermark1 = 1;
    (void)static_cast<int>(send_low_watermark1.value());

    boost::asio::socket_base::receive_buffer_size receive_buffer_size1(128);
    sk.set_option(receive_buffer_size1);
    boost::asio::socket_base::receive_buffer_size receive_buffer_size2;
    sk.get_option(receive_buffer_size2);
    receive_buffer_size1 = 1;
    (void)static_cast<int>(receive_buffer_size1.value());

    boost::asio::socket_base::receive_low_watermark receive_low_watermark1(128);
    sk.set_option(receive_low_watermark1);
    boost::asio::socket_base::receive_low_watermark receive_low_watermark2;
    sk.get_option(receive_low_watermark2);
    receive_low_watermark1 = 1;
    (void)static_cast<int>(receive_low_watermark1.value());

    boost::asio::socket_base::reuse_address reuse_address1(true);
    sk.set_option(reuse_address1);
    boost::asio::socket_base::reuse_address reuse_address2;
    sk.get_option(reuse_address2);
    reuse_address1 = true;
    (void)static_cast<bool>(reuse_address1);
    (void)static_cast<bool>(!reuse_address1);
    (void)static_cast<bool>(reuse_address1.value());

    boost::asio::socket_base::linger linger1(true, 30);
    sk.set_option(linger1);
    boost::asio::socket_base::linger linger2;
    sk.get_option(linger2);
    linger1.enabled(true);
    linger1.timeout(1);
    (void)static_cast<bool>(linger1.enabled());
    (void)static_cast<bool>(linger1.timeout());

    boost::asio::socket_base::enable_connection_aborted
      enable_connection_aborted1(true);
    sk.set_option(enable_connection_aborted1);
    boost::asio::socket_base::enable_connection_aborted
      enable_connection_aborted2;
    sk.get_option(enable_connection_aborted2);
    enable_connection_aborted1 = true;
    (void)static_cast<bool>(enable_connection_aborted1);
    (void)static_cast<bool>(!enable_connection_aborted1);
    (void)static_cast<bool>(enable_connection_aborted1.value());

    boost::asio::socket_base::non_blocking_io non_blocking_io(true);
    sk.io_control(non_blocking_io);

    boost::asio::socket_base::bytes_readable bytes_readable;
    sk.io_control(bytes_readable);
    std::size_t bytes = bytes_readable.get();
    (void)bytes;

  } catch (std::exception&) {}
}
} // namespace

namespace runtime {
void test_socket_base() {
  boost::asio::io_service ios;
  boost::asio::ip::udp::socket udp_sk(ios, boost::asio::ip::udp::v4());
  boost::asio::ip::tcp::socket tcp_sk(ios, boost::asio::ip::tcp::v4());
  boost::asio::ip::tcp::acceptor ap(ios, boost::asio::ip::tcp::v4());
  boost::system::error_code ec;

  boost::asio::socket_base::broadcast broadcast1(true);
  assert(broadcast1.value());
  assert(static_cast<bool>(broadcast1));
  assert(!!broadcast1);
  udp_sk.set_option(broadcast1, ec);
  assert(!ec);

  boost::asio::socket_base::broadcast broadcast2;
  udp_sk.get_option(broadcast2, ec);
  assert(!ec);
  assert(broadcast2.value());
  assert(static_cast<bool>(broadcast2));
  assert(!!broadcast2);

  boost::asio::socket_base::broadcast broadcast3(false);
  assert(!broadcast3.value());
  assert(!static_cast<bool>(broadcast3));
  assert(!broadcast3);
  udp_sk.set_option(broadcast3, ec);
  assert(!ec);

  boost::asio::socket_base::broadcast broadcast4;
  udp_sk.get_option(broadcast4, ec);
  assert(!ec);
  assert(!broadcast4.value());
  assert(!static_cast<bool>(broadcast4));
  assert(!broadcast4); 

  boost::asio::socket_base::debug debug1(true);
  assert(debug1.value());
  assert(static_cast<bool>(debug1));
  assert(!!debug1);
  udp_sk.set_option(debug1, ec);
#if defined(__linux__)
  bool not_root = (ec == boost::asio::error::access_denied);
  assert(!ec || not_root);
#endif
  boost::asio::socket_base::debug debug2;
  udp_sk.get_option(debug2, ec);
#if defined(__linux__)
  assert(debug2.value() || not_root);
  assert(static_cast<bool>(debug2) || not_root);
  assert(!!debug2 || not_root);
#else
  assert(debug2.value());
  assert(static_cast<bool>(debug2));
  assert(!!debug2);
#endif

  boost::asio::socket_base::debug debug4;
  udp_sk.get_option(debug4, ec);
#if defined(__linux__)
  assert(!debug4.value() || not_root);
  assert(!static_cast<bool>(debug4) || not_root);
  assert(!debug4 || not_root);
#else
  assert(!debug4.value());
  assert(!static_cast<bool>(debug4));
  assert(!debug4);
#endif

  boost::asio::socket_base::do_not_route do_not_route1(true);
  assert(do_not_route1.value());
  assert(static_cast<bool>(do_not_route1));
  assert(!!static_cast<bool>(do_not_route1));
  assert(!!do_not_route1);
  udp_sk.set_option(do_not_route1, ec);
  assert(!ec);

  boost::asio::socket_base::do_not_route do_not_route2;
  udp_sk.get_option(do_not_route2, ec);
  assert(!ec);
  assert(do_not_route2.value());
  assert(static_cast<bool>(do_not_route2));
  assert(!!static_cast<bool>(do_not_route2));
  assert(!!do_not_route2);

  boost::asio::socket_base::do_not_route do_not_route3(false);
  assert(!do_not_route3.value());
  assert(!static_cast<bool>(do_not_route3));
  assert(!do_not_route3);
  udp_sk.set_option(do_not_route3, ec);
  assert(!ec);

  boost::asio::socket_base::do_not_route do_not_route4;
  udp_sk.get_option(do_not_route4, ec);
  assert(!ec);
  assert(!do_not_route4.value());
  assert(!static_cast<bool>(do_not_route4));
  assert(!do_not_route4);

  boost::asio::socket_base::keep_alive keep_alive1(true);
  tcp_sk.set_option(keep_alive1, ec);
  assert(keep_alive1.value());
  assert(static_cast<bool>(keep_alive1));
  assert(!!static_cast<bool>(keep_alive1));
  assert(!!keep_alive1);
  assert(!ec);

  boost::asio::socket_base::keep_alive keep_alive2;
  tcp_sk.get_option(keep_alive2, ec);
  assert(!ec);
  assert(keep_alive2.value());
  assert(static_cast<bool>(keep_alive2));
  assert(!!static_cast<bool>(keep_alive2));
  assert(!!keep_alive2);

  boost::asio::socket_base::keep_alive keep_alive3(false);
  tcp_sk.set_option(keep_alive3, ec);
  assert(!keep_alive3.value());
  assert(!static_cast<bool>(keep_alive3));
  assert(!keep_alive3);
  assert(!ec);

  boost::asio::socket_base::keep_alive keep_alive4;
  tcp_sk.get_option(keep_alive4, ec);
  assert(!ec);
  assert(!keep_alive4.value());
  assert(!static_cast<bool>(keep_alive4));
  assert(!keep_alive4);

  boost::asio::socket_base::send_buffer_size send_buffer_size1(4096*2);
  tcp_sk.set_option(send_buffer_size1, ec);
  assert(send_buffer_size1.value() == 4096*2);
  assert(!ec);

  boost::asio::socket_base::send_buffer_size send_buffer_size2;
  tcp_sk.get_option(send_buffer_size2, ec);
  assert(!ec);
  assert(send_buffer_size2.value() == 4096*2);

  boost::asio::socket_base::send_buffer_size send_buffer_size3(
    send_buffer_size2.value() * 2);
  tcp_sk.set_option(send_buffer_size3, ec);
  assert(!ec);
  assert(send_buffer_size3.value() == 4096*2*2);

  boost::asio::socket_base::send_buffer_size send_buffer_size4;
  tcp_sk.get_option(send_buffer_size4, ec);
  assert(!ec);
  assert(send_buffer_size4.value() == send_buffer_size1.value() * 2);

  boost::asio::socket_base::send_low_watermark send_low_watermark1(4096);
  tcp_sk.set_option(send_low_watermark1, ec);
  assert(send_low_watermark1.value() == 4096);
#if defined(WIN32) || defined(__linux__) || defined(__sun)
  assert(!!ec);
#else
  assert(!ec);
#endif

  boost::asio::socket_base::send_low_watermark send_low_watermark2;
  tcp_sk.get_option(send_low_watermark2, ec);
#if defined(WIN32) || defined(__linux__) || defined(__sun)
  assert(!ec);
#else
  assert(!!ec);
  assert(send_low_watermark2.value() == 4096);
#endif

  boost::asio::socket_base::send_low_watermark send_low_watermark3(
    send_low_watermark2.value() * 2);
  tcp_sk.set_option(send_low_watermark3, ec);
#if defined(WIN32) || defined(__linux__) || defined(__sun)
  assert(!!ec);
#else
  assert(send_low_watermark3.value() == 4096*2);
#endif

  boost::asio::socket_base::send_low_watermark send_low_watermark4;
  tcp_sk.get_option(send_low_watermark4, ec);
#if defined(WIN32) || defined(__sun)
  assert(!!ec);
#elif defined(__linux__)
  assert(!ec);
#else
  assert(send_low_watermark4.value() == 4096*2);
#endif

  boost::asio::socket_base::receive_buffer_size receive_buffer_size1(4096);
  tcp_sk.set_option(receive_buffer_size1, ec);
  assert(!ec);
  assert(receive_buffer_size1.value() == 4096);

  boost::asio::socket_base::receive_buffer_size receive_buffer_size2;
  tcp_sk.get_option(receive_buffer_size2, ec);
  assert(!ec);
  assert(receive_buffer_size2.value() == receive_buffer_size1.value());

  boost::asio::socket_base::receive_buffer_size receive_buffer_size3(
    receive_buffer_size1.value()*2);
  tcp_sk.set_option(receive_buffer_size3, ec);
  assert(!ec);
  assert(receive_buffer_size3.value() == receive_buffer_size2.value()*2);

  boost::asio::socket_base::receive_buffer_size receive_buffer_size4;
  tcp_sk.get_option(receive_buffer_size4, ec);
  assert(!ec);
  assert(receive_buffer_size4.value() == 4096*2);

  boost::asio::socket_base::receive_low_watermark receive_low_watermark1(4096);
  tcp_sk.set_option(receive_low_watermark1, ec);
  assert(!ec);
  assert(receive_low_watermark1.value() == 4096);

  boost::asio::socket_base::receive_low_watermark receive_low_watermark2;
  tcp_sk.get_option(receive_low_watermark2, ec);
  assert(!ec);
  assert(receive_low_watermark2.value() == receive_low_watermark1.value());

  boost::asio::socket_base::receive_low_watermark receive_low_watermark3(
    receive_low_watermark2.value()*2);
  tcp_sk.set_option(receive_low_watermark3, ec);
  assert(!ec);
  assert(receive_low_watermark3.value() == 4096*2);

  boost::asio::socket_base::receive_low_watermark receive_low_watermark4;
  tcp_sk.get_option(receive_low_watermark4, ec);
  assert(!ec);
  assert(receive_low_watermark4.value() == receive_low_watermark2.value()*2);
}
} // namespace
auto main() -> decltype(0) {
  compile::test_socket_base();
  runtime::test_socket_base();
  return 0;
}
