#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "yield.hpp"

class tcp_server : coroutine {
public:
  tcp_server(boost::asio::ip::tcp::acceptor& ap, std::size_t buffer_size)
    : ap_(ap), sk_(ap_.get_io_service()), buffer_(buffer_size) {}

  void operator()(boost::system::error_code ec, std::size_t n = 0) {
    reenter(this) for (;;) {
      yield ap_.async_accept(sk_, ref(this));

      while (!ec) {
        yield boost::asio::async_read(sk_, boost::asio::buffer(buffer_),
          ref(this));

        if (!ec) 
          for (std::size_t i = 0; i < n; ++i)
            buffer_[i] = ~buffer_[i];

          yield boost::asio::async_write(sk_, boost::asio::buffer(buffer_),
            ref(this));
      } 
      sk_.close();
    }
  }

  struct ref {
    explicit ref(tcp_server* p) : p_(p) {}

    void operator()(boost::system::error_code ec, std::size_t n = 0) {
      (*p_)(ec, n);
    }

  private:
    tcp_server* p_;
  };

private:
  boost::asio::ip::tcp::acceptor& ap_;
  boost::asio::ip::tcp::socket sk_;
  std::vector<unsigned char> buffer_;
  boost::asio::ip::tcp::endpoint sender_;
};

#include "unyield.hpp"

auto main() -> decltype(0) {
  unsigned short port = 12345;
  int max_connection = 100;
  std::size_t buffer_size = 1024;
  bool spin = true;

  boost::asio::io_service ios;
  boost::asio::ip::tcp::acceptor ap(ios, boost::asio::ip::tcp::endpoint(
    boost::asio::ip::tcp::v4(), port));
  std::vector<boost::shared_ptr<tcp_server> > servers;

  for (int i = 0; i < max_connection; ++i) {
    boost::shared_ptr<tcp_server> s(new tcp_server(ap, buffer_size));
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
