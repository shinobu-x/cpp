#include "../hpp/basic_raw_protocol.hpp"

#include <boost/asio/io_service.hpp>
#include <boost/asio/buffer.hpp>
#include <boost/asio/write.hpp>
#include <boost/system/error_code.hpp>

#include <chrono>
#include <cstdlib> /* size_t */
#include <iostream>
#include <thread>
#include <utility>

typedef basic_raw_protocol<AF_INET, AF_INET6, SOCK_RAW, IPPROTO_TCP> tcp_type;

class session;
class server;

class session
 : public std::enable_shared_from_this<session> {
public:
  session(tcp_type::socket sk)
    : sk_(std::move(sk_)) {
    std::cout << "Connected..." << '\n';
  }

  void do_read() {
    read_();
  }
private:
  void read_() {
    std::cout << "Reading..." << '\n';
    auto self(shared_from_this());
    sk_.async_read_some(
      boost::asio::buffer(data_, max_length_),
      [this, self] (boost::system::error_code ec, size_t length) {
        if (!ec)
          write_(length);
      }
    );
  }

  void write_(size_t length) {
    std::cout << "Writing..." << '\n';
    auto self(shared_from_this());
    boost::asio::async_write(sk_,
      boost::asio::buffer(data_, length),
      [this, self] (boost::system::error_code ec, size_t length) {
        if (!ec)
          write_(length);
      }
    );
  }

  tcp_type::socket sk_;
  enum { max_length_ = 1024 };
  char data_[max_length_];
};

class server {
public:
  server(boost::asio::io_service& ios, short port)
    : ap_(ios, tcp_type::endpoint(tcp_type::v4(), port)),
      sk_(ios, tcp_type::v4()) {
    std::cout << "Listening..." << '\n';
    accept_();
  }

private:
  void accept_() {
    ap_.async_accept(sk_,
      [this](boost::system::error_code ec) {
        if (!ec)
          std::make_shared<session>(std::move(sk_))->do_read();
        accept_();
      }
    );
  }

  tcp_type::acceptor ap_;
  tcp_type::socket sk_;
};

void doit() {
  boost::asio::io_service ios;
  short port = 12345;
  server s(ios, port);
  ios.run();
} 
auto main() -> decltype(0)
{
  doit();
  return 0;
}
