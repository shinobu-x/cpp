#include <boost/asio/io_service.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>
#include <boost/asio/detail/shared_ptr.hpp>

#include <cstdio>
#include <cstdlib> /* size_t */
#include <cstring>
#include <vector>

#include "../hpp/tcp.hpp"
#include "../hpp/yield.hpp"

class tcp_server : coroutine {
public:
  tcp_server(tcp::acceptor& acceptor, std::size_t buf_size)
    : acceptor_(acceptor),
      socket_(acceptor_.get_io_service()),
      buffer_(buf_size) {}

  void operator() (boost::system::error_code ec, std::size_t n = 0) {
    reenter (this) for (;;) {
      yield acceptor_.async_accept(socket_, ref(this));

      while (!ec) {
        yield boost::asio::async_read(socket_,
          boost::asio::buffer(buffer_), ref(this));

        if (!ec) {
          for (std::size_t i = 0; i < n; ++i)
            buffer_[i] = ~buffer_[1];

          yield boost::asio::async_write(socket_,
            boost::asio::buffer(buffer_), ref(this));
        }
      }

      socket_.close();
    }
  }

  struct ref {
    explicit ref(tcp_server* p)
      : p_(p) {}

    void operator() (boost::system::error_code ec, std::size_t n = 0) {
      (*p_)(ec, n);
     }

  private:
    tcp_server* p_;
  };

private:
  tcp::acceptor& acceptor_;
  tcp::socket socket_;
  std::vector<unsigned char> buffer_;
  tcp::endpoint sender_;
};

#include "../hpp/unyield.hpp"

auto main() -> decltype(0)
{
  unsigned short port = 12345;
  int max_connections = 3;
  std::size_t buf_size = 1024;
  bool spin = true;

  boost::asio::io_service ios;
  tcp::acceptor acceptor(ios, tcp::endpoint(tcp::v4(), port));
  std::vector<boost::asio::detail::shared_ptr<tcp_server> > servers;

  for (int i = 0; i < max_connections; ++i) {
    boost::asio::detail::shared_ptr<tcp_server> s(new tcp_server(
      acceptor, buf_size));
    servers.push_back(s);
    (*s)(boost::system::error_code());
  }

  if (spin)
    for (;;)
      ios.poll();
  else
    ios.run();

  return 0;
}
