#include <boost/asio/ip/unicast.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/udp.hpp>

#include <cassert>

void test_1() {
  try {
    boost::asio::io_service ios;
    boost::asio::ip::udp::socket s1(ios);

    boost::asio::ip::unicast::hops hops1(1024);
    s1.set_option(hops1);
    boost::asio::ip::unicast::hops hops2;
    s1.get_option(hops2);
    hops1 = 1;
    (void)static_cast<int>(hops1.value());
  } catch (std::exception&) {}
}

void test_2() {
  boost::asio::io_service ios;
  boost::system::error_code ec;
  boost::asio::ip::udp::endpoint ep_v4(
    boost::asio::ip::address_v4::loopback(), 0);
  boost::asio::ip::udp::socket s1(ios);
  s1.open(ep_v4.protocol(), ec);
  s1.bind(ep_v4,ec);
  bool have_v4 = !ec;

  boost::asio::ip::udp::endpoint ep_v6(
    boost::asio::ip::address_v6::loopback(), 0);
  boost::asio::ip::udp::socket s2(ios);
  s2.open(ep_v6.protocol(), ec);
  s2.bind(ep_v6, ec);
  bool have_v6 = !ec;

  assert(have_v4 || have_v6);

  if (have_v4) {
    boost::asio::ip::unicast::hops hops1(1);
    assert(hops1.value() == 1);
    s1.set_option(hops1, ec);
#if defined(BOOST_ASIO_WINDOWS) && defined(UNDER_CE)
    assert(ec == boost::asio::error::no_protocol_option);
#else
    assert(!ec);
#endif
    boost::asio::ip::unicast::hops hops2;
    s1.get_option(hops2, ec);
#if defined(BOOST_ASIO_WINDOWS) && defined(UNDER_CE)
    assert(ec == boost::asio::error::no_protocol_option);
#else
    assert(!ec);
    assert(hops2.value() == 1);
#endif
    boost::asio::ip::unicast::hops hops3(255);
    assert(hops3.value() == 255);
    s1.set_option(hops3, ec);
#if defined(BOOST_ASIO_WINDOWS) && defined(UNDER_CE)
    assert(ec == boost::asio::error::no_protocol_option);
#else
    assert(!ec);
#endif
    boost::asio::ip::unicast::hops hops4;
    s1.get_option(hops4, ec);
#if defined(BOOST_ASIO_WINDOWS) && defined(UNDER_CE)
    assert(ec == boost::asio::error::no_protocol_option);
#else
    assert(!ec);
    assert(hops4.value() == 255);
#endif
  }

  if (have_v6) {
    boost::asio::ip::unicast::hops hops1(1);
    assert(hops1.value() == 1);
    s2.set_option(hops1, ec);
    assert(!ec);
    boost::asio::ip::unicast::hops hops2;
    s2.get_option(hops2, ec);
    assert(!ec);
    assert(hops2.value() == 1);
    boost::asio::ip::unicast::hops hops3(255);
    assert(hops3.value() == 255);
    s2.set_option(hops3, ec);
    assert(!ec);
    boost::asio::ip::unicast::hops hops4;
    s2.get_option(hops4, ec);
    assert(!ec);
    assert(hops4.value() == 255);
  }
}

auto main() -> decltype(0) {
  test_1(); test_2();
  return 0;
}  
