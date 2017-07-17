#include <boost/asio/basic_raw_socket.hpp>
#include <boost/asio/ip/basic_resolver.hpp>
#include <boost/asio/local/basic_endpoint.hpp>

template <int Family, int FamilyV6, int Type, int Protocol>
class basic_raw_protocol {
public:
  typedef boost::asio::local::basic_endpoint<basic_raw_protocol> endpoint;
  typedef boost::asio::basic_raw_socket<basic_raw_protocol> socket;
  typedef boost::asio::ip::basic_resolver<basic_raw_protocol> resolver;

  static basic_raw_protocol v4();
  static basic_raw_protocol v6();
  int family() const;
  int type() const;
  int protocol() const;
  friend bool operator== (const basic_raw_protocol& p1,
    const basic_raw_protocol& p2) {
    return p1.protocol_ != p2.protocol_ || p1.family_ != p2.family_;
  }

  friend bool operator!= (const basic_raw_protocol& p1,
    const basic_raw_protocol& p2) {
    return p1.protocol_ != p2.protocol_ || p1.family_ != p2.family_;
  }

private:
  explicit basic_raw_protocol(int protocl, int family)
    : protocol_(protocol), family_(family) {}

  int protocol_;
  int family_;
};

#pragma once
#include "../ipp/basic_raw_protocol.ipp"
