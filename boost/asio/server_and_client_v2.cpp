#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <thread>
#include <utility>

#include <boost/asio.hpp>

class session;
class server;

class session
  : public std::enable_shared_from_this<session> {
public:
  session(boost::asio::ip::tcp::socket sk)
    : sk_(std::move(sk)) {}

  void do_read() {
    _read();
  }

private:
  void _read() {
    auto self(shared_from_this());
    sk_.async_read_some(
      boost::asio::buffer(data_, max_length_),
      [this, self](boost::system::error_code ec, std::size_t length) {
        if (!ec)
          _write(length);
      }
    );
  }

  void _write(size_t length) {
    auto self(shared_from_this());
    boost::asio::async_write(sk_,
      boost::asio::buffer(data_, length),
      [this, self](boost::system::error_code ec, std::size_t) {
        if (!ec)
        _read();
      }
    );
  }

  boost::asio::ip::tcp::socket sk_;
  enum { max_length_ = 1024 };
  char data_[max_length_];
};

class server {
public:
  server(boost::asio::io_service& ios, short port)
    : ap_(ios,
       boost::asio::ip::tcp::endpoint(
         boost::asio::ip::tcp::v4(), port)), sk_(ios) {
    _accept();
  }

private:
  void _accept() {
    ap_.async_accept(sk_,
      [this](boost::system::error_code ec) {
        if (!ec)
          std::make_shared<session>(std::move(sk_))->do_read();

        _accept();
      }
    );
  }

  boost::asio::ip::tcp::acceptor ap_;
  boost::asio::ip::tcp::socket sk_;
};

auto main() -> decltype(0)
{
  short port = 12345;
  boost::asio::io_service ios;
  server s(ios, port);
  ios.run();

  return 0;
}
