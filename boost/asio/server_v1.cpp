#include <cstdlib>
#include <iostream>
#include <thread>
#include <utility>

#include <boost/asio.hpp>

void session(boost::asio::ip::tcp::socket sk) {
  try {
    for (;;) {
      char data[1024];

      boost::system::error_code e;
      size_t length = sk.read_some(
        boost::asio::buffer(data), e);

      if (e == boost::asio::error::eof) break; // Closed by peer.
      else if (e)
        throw boost::system::system_error(e); // Mmm...

      boost::asio::write(sk, boost::asio::buffer(data, length));
    }
  } catch (std::exception& e) {
    std::cerr << "Exception: " << e.what() << '\n';
  }
}

void server(boost::asio::io_service& ios, unsigned short port) {

  boost::asio::ip::tcp::acceptor ap(ios,
    boost::asio::ip::tcp::endpoint(
      boost::asio::ip::tcp::v4(), port));

  for (;;) {
    boost::asio::ip::tcp::socket sk(ios);
    ap.accept(sk);
    std::thread(session, std::move(sk)).detach();
  }
}

auto main() -> decltype(0)
{
  boost::asio::io_service ios;
  unsigned short port = 12345;
  server(ios, port);
  return 0;
}
