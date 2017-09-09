#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/udp.hpp>
#include <boost/shared_ptr.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "allocator.hpp"
#include "yield.hpp"

class udp_server : coroutine {
public:
  udp_server(boost::asio::io_service& ios, unsigned short port,
    std::size_t buffer_size)
    : sk_(ios, boost::asio::ip::udp::endpoint(
        boost::asio::ip::udp::v4(), port)),  buffer_(buffer_size) {}

  void operator()(boost::system::error_code ec, std::size_t n = 0) {
    reenter(this) for (;;) {
      yield sk_.async_receive_from(boost::asio::buffer(buffer_),
        ep_, ref(this));

      if (!ec)
        for (std::size_t i = 0; i < n; ++i)
          buffer_[i] = ~buffer_[i];
        sk_.send_to(boost::asio::buffer(buffer_, n), ep_, 0, ec);
    }
  }

  friend void* asio_handler_allocate(std::size_t n, udp_server* s) {
    return s->allocator_.allocate(n);
  }

  friend void asio_handler_deallocate(void* p, std::size_t, udp_server* s) {
    s->allocator_.deallocate(p);
  }

  struct ref {
    explicit ref(udp_server* p) : p_(p) {}

    void operator()(boost::system::error_code ec, std::size_t n = 0) {
      (*p_)(ec, n);
    }

  private:
    udp_server* p_;

    friend void* asio_handler_allocate(std::size_t n, ref* r) {
      return asio_handler_allocate(n, r->p_);
    }

    friend void asio_handler_deallocate(void* p, std::size_t n, ref* r) {
      asio_handler_deallocate(p, n, r->p_);
    }
  };

private:
  boost::asio::ip::udp::socket sk_;
  std::vector<unsigned char> buffer_;
  boost::asio::ip::udp::endpoint ep_;
  allocator allocator_;
};

#include "unyield.hpp"

auto main() -> decltype(0) {
  unsigned short first_port = 12345;
  unsigned short num_ports = 1000;
  std::size_t buffer_size = 1024;
  bool spin = true;

  boost::asio::io_service ios;
  std::vector<boost::shared_ptr<udp_server> > servers;
  for (unsigned short i = 0; i < num_ports; ++i) {
    unsigned short port = first_port + i;
    boost::shared_ptr<udp_server> s(new udp_server(ios, port, buffer_size));
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
