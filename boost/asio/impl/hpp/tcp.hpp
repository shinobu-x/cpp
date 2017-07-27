#include <boost/asio/detail/config.hpp>
#include <boost/asio/basic_socket_acceptor.hpp>
#include <boost/asio/basic_socket_iostream.hpp>
#include <boost/asio/basic_stream_socket.hpp>
#include <boost/asio/detail/socket_option.hpp>
#include <boost/asio/detail/socket_types.hpp>
#include <boost/asio/ip/basic_endpoint.hpp>
#include <boost/asio/ip/basic_resolver.hpp>
#include <boost/asio/ip/basic_resolver_iterator.hpp>
#include <boost/asio/ip/basic_resolver_query.hpp>

class tcp {
public:
  typedef boost::asio::ip::basic_endpoint<tcp> endpoint;
  typedef boost::asio::basic_stream_socket<tcp> socket;
  typedef boost::asio::basic_socket_acceptor<tcp> acceptor;
  typedef boost::asio::ip::basic_resolver<tcp> resolver;
  typedef boost::asio::basic_socket_iostream<tcp> iostream;

  typedef boost::asio::detail::socket_option::boolean<
    IPPROTO_TCP, TCP_NODELAY> no_delay;

  static tcp v4() { return tcp(PF_INET); }
  static tcp v6() { return tcp(PF_INET6); }
  int type() const { return SOCK_STREAM; }
  int protocol() const { return IPPROTO_TCP; }
  int family() const { return family_; }

  friend bool operator== (const tcp& p1, const tcp& p2) {
    return p1.family_ == p2.family_;
  }

  friend bool operator!= (const tcp& p1, const tcp& p2) {
    return p1.family_ != p2.family_;
  }
private:
  explicit tcp(int protocol_family)
    : family_(protocol_family) {}

  int family_;
};
