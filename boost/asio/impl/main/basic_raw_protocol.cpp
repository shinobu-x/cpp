#include "../hpp/basic_raw_protocol.hpp"

#include <boost/system/error_code.hpp>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <utility>

template <typename T>
void rs_handler(const boost::system::error_code&, T) {}

void doit() {
  typedef basic_raw_protocol<AF_INET, AF_INET6, SOCK_RAW, IPPROTO_TCP> tcp_type;
 
  boost::system::error_code ec; 
  boost::asio::io_service ios;

  tcp_type::socket sk1(ios);
  tcp_type::socket sk2(ios, tcp_type::v4());
  tcp_type::socket sk3(ios, tcp_type::v6());
  tcp_type::socket sk4(ios, tcp_type::endpoint(tcp_type::v4(), 0));
  tcp_type::socket sk5(ios, tcp_type::endpoint(tcp_type::v6(), 0));

  tcp_type::socket::native_handle_type native_socket1 =
    ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);

  tcp_type::socket sk6(ios, tcp_type::v4(), native_socket1);
  tcp_type::socket sk7(std::move(sk6));

  sk1 = tcp_type::socket(ios);
  sk1 = std::move(sk2);

  boost::asio::io_service& ref = sk1.get_io_service();

  tcp_type::socket::lowest_layer_type& ll1 = sk1.lowest_layer();
  (void)ll1;

  const tcp_type::socket& sk8 = sk1;
  const tcp_type::socket::lowest_layer_type& ll2  = sk8.lowest_layer();
  (void)ll2;

  assert(sk1.is_open());
  sk1.close();
  assert(!sk1.is_open());
  sk1.open(tcp_type::v4());
  sk1.close();
  assert(!sk1.is_open());
  sk1.open(tcp_type::v6());
  sk1.close();
  assert(!sk1.is_open());
  sk1.open(tcp_type::v4(), ec);
  sk1.close(ec);
  assert(!sk1.is_open());
  sk1.open(tcp_type::v6(), ec);
  sk1.close(ec);
  assert(!sk1.is_open());

  tcp_type::socket::native_handle_type native_socket2 =
    ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  sk1.assign(tcp_type::v4(), native_socket2);
  assert(sk1.is_open());
  sk1.close();
  assert(!sk1.is_open());

  tcp_type::socket::native_handle_type native_socket3 =
    ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  sk1.assign(tcp_type::v4(), native_socket3, ec);
  assert(sk1.is_open());
  sk1.cancel();
  sk1.cancel(ec);
  assert(sk1.is_open());
  sk1.close();
}
auto main() -> decltype(0)
{
  doit();
  return 0;
}
