#include "hpp/basic_raw_protocol.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>

#include <sys/socket.h>

typedef basic_raw_protocol<AF_INET, AF_INET6, SOCK_RAW, IPPROTO_TCP> raw_tcp;

constexpr std::uint16_t hton(std::uint16_t s) {
  return (s >> 8) | (s << 8);
}

auto main() -> decltype(0)
{
  typedef basic_raw_protocol<AF_INET, AF_INET6, SOCK_RAW, IPPROTO_TCP> tcp_type;
  
  raw_tcp v4_t = tcp_type::v4();
  raw_tcp v6_t = tcp_type::v6();
  int v4_family = v4_t.family();
  int v6_family = v6_t.family();
  int v4_type = v4_t.type();
  int v6_type = v6_t.type();
  int v4_protocol = v4_t.protocol();
  int v6_protocol = v6_t.protocol();
  assert(v4_t == v4_t);
  assert(v6_t == v6_t); 
  assert(v4_t != v6_t);
  return 0;
}
