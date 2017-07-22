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
  
  boost::asio::io_service ios;
  tcp_type::socket sk1(ios);
  tcp_type::socket sk2(ios, tcp_type::v4());
  tcp_type::socket sk3(ios, tcp_type::v6());
  tcp_type::socket sk4(ios, tcp_type::endpoint(tcp_type::v4(), 0));
  tcp_type::socket sk5(ios, tcp_type::endpoint(tcp_type::v6(), 0));
  tcp_type::socket::native_handle_type native_socket1;
  tcp_type::socket sk6(ios, tcp_type::v4(), native_socket1);
  tcp_type::socket sk7(std::move(sk6));
}
auto main() -> decltype(0)
{
  doit();
  return 0;
}
