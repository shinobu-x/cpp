#include <boost/asio/ip/v6_only.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ip/udp.hpp>

#include <cassert>

void test_1() {
  try {
    boost::asio::io_service ios;
    boost::asio::ip::udp::socket s1(ios);

    boost::asio::ip::v6_only v6_only1(true);
    s1.set_option(v6_only1);
    boost::asio::ip::v6_only v6_only2;
    s1.get_option(v6_only2);
    v6_only1 = true;
    (void)static_cast<bool>(v6_only1);
    (void)static_cast<bool>(!v6_only1);
    (void)static_cast<bool>(v6_only1.value());
  } catch (std::exception&) {}
}

void test_2() {
  boost::asio::io_service ios;
  boost::system::error_code ec;
  boost::asio::ip::tcp::endpoint ep_v6(
    boost::asio::ip::address_v6::loopback(), 0);
  boost::asio::ip::tcp::acceptor ap_v6(ios);
  ap_v6.open(ep_v6.protocol(), ec);
  ap_v6.bind(ep_v6, ec);
  bool have_v6 = !ec;
  ap_v6.close(ec);
  ap_v6.open(ep_v6.protocol(), ec);

  if (have_v6) {
    boost::asio::ip::v6_only v6_only2(false);
    assert(!v6_only2.value());
    assert(!static_cast<bool>(v6_only2));
    assert(!v6_only2);
    ap_v6.set_option(v6_only2, ec);
    assert(!ec);

    boost::asio::ip::v6_only v6_only3;
    ap_v6.get_option(v6_only3, ec);
    assert(!ec);
    assert(!v6_only3.value());
    assert(!static_cast<bool>(v6_only3));
    assert(!v6_only3);

    boost::asio::ip::v6_only v6_only4(true);
    assert(v6_only4.value());
    assert(static_cast<bool>(v6_only4));
    assert(!!v6_only4);
    ap_v6.set_option(v6_only4, ec);
    assert(!ec);

    boost::asio::ip::v6_only v6_only5;
    ap_v6.get_option(v6_only5, ec);
    assert(!ec);
    assert(v6_only5.value());
    assert(static_cast<bool>(v6_only5));
    assert(!!v6_only5);
  }
}

auto main() -> decltype(0) {
  test_1(); test_2();
  return 0;
}
