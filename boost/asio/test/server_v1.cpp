#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <utility>

#include <boost/asio.hpp>

void session(boost::asio::ip::tcp::socket sk) {

  std::cout << "Session started..." << '\n';

  for (;;) {

    std::cout << "Connected..." << '\n';

    char data[1024] = {"abc"};
    boost::system::error_code ec;
    size_t len = sk.read_some(boost::asio::buffer(data), ec);
    if (ec == boost::asio::error::eof)
      break;
    else if (ec)
      throw boost::system::system_error(ec); // Mmm...

    boost::asio::write(sk, boost::asio::buffer(data, len));
  }
}

void server(boost::asio::io_service& ios, unsigned short port) {

  std::cout << "Listening..." << '\n';

  boost::asio::ip::tcp::acceptor ap(ios,
    boost::asio::ip::tcp::endpoint(
      boost::asio::ip::tcp::v4(), port));

  for (;;) {
    boost::asio::ip::tcp::socket sk(ios);
    ap.accept(sk);
    std::thread(session, std::move(sk)).detach();
  }
}

void client() {

  std::this_thread::sleep_for(std::chrono::seconds(3));

  boost::asio::io_service ios;
  boost::asio::ip::tcp::socket sk(ios);
  boost::asio::ip::tcp::resolver rs(ios);
  boost::asio::connect(sk, rs.resolve({"localhost", "12345"}));

  char data[1024] = "abc";
  size_t len = std::strlen(data);
  boost::asio::write(sk, boost::asio::buffer(data, len));

  char res[1024];
  len = boost::asio::read(sk, boost::asio::buffer(res, len));
  std::cout.write(res, len);
  std::cout << '\n';
}

auto main() -> decltype(0)
{
  std::thread s([]{
    unsigned short port = 12345;
    boost::asio::io_service ios;
    server(ios, port);
  });

  s.detach();

  // ******

  std::thread c(client);

  c.join();
  return 0;
}
