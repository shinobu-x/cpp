#include <boost/asio/basic_socket_acceptor.hpp>
#include <boost/asio/basic_socket_iostream.hpp>
#include <boost/asio/basic_stream_socket.hpp>

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/socket_option.hpp>
#include <boost/asio/detail/socket_types.hpp>

#include <boost/asio/ip/basic_endpoint.hpp>
#include <boost/asio/ip/basic_resolver.hpp>
#include <boost/asio/ip/basic_resolver_iterator.hpp>
#include <boost/asio/ip/basic_resolver_query.hpp>

template <int Domain, int DomainV6, int Type, int Protocol>
class basic_raw_protocol {
public:
  typedef boost::asio::ip::basic_endpoint<basic_raw_protocol> endpoint;
  typedef boost::asio::basic_stream_socket<basic_raw_protocol> socket;
  typedef boost::asio::basic_socket_acceptor<basic_raw_protocol> acceptor;
  typedef boost::asio::ip::basic_resolver<basic_raw_protocol> resolver;
  typedef boost::asio::basic_socket_iostream<basic_raw_protocol> iostream;

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
  explicit basic_raw_protocol(int, int);

  int protocol_;
  int family_;
  int type_;
};

#pragma once
#include "../ipp/basic_raw_protocol.ipp"
