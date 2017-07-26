#include "../hpp/spawn.hpp"

#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/asio/strand.hpp>

#include <boost/bind.hpp>
#include <boost/system/error_code.hpp>

class A {
public:
 A() : ios_(), sk_(ios_) {}

 int do_some() {
    boost::system::error_code ec;
//    spawn(ios_, boost::bind(&A::doing, this, _1));
    ios_.run(ec);

    if (ec)
      return EXIT_FAILURE;
    return EXIT_SUCCESS;
  }

  void doing(yield_context yld) {
    boost::system::error_code ec;
  }
private:
  boost::asio::io_service ios_;
  boost::asio::ip::tcp::socket sk_;
};

auto main() -> decltype(0)
{
  boost::asio::io_service ios;
  boost::asio::io_service::strand std(ios);

  return 0;
}
