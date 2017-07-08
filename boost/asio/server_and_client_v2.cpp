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
    : sk_(std::move(sk)) {
    std::cout << "Connected..." << '\n';
  }

  void do_read() {
    _read();
  }

private:
  void _read() {
    std::cout << "Reading..." << '\n';
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
    std::cout << "Writing..." << '\n';
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
    std::cout << "Listening..." << '\n';
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

class client {
public:
  client(boost::asio::io_service& ios, const std::string& host, const std::string& port) 
    : rs_(ios), sk_(ios), host_(host), port_(port) {
  }

  void do_connect() {
    connect_();
  }
    
  void do_write(const char* data) {
    len_ = std::strlen(data);
    write_(data);
  }

  void do_read() {
    read_();
  }

private:
  boost::asio::ip::tcp::socket sk_;
  boost::asio::ip::tcp::resolver rs_;
  size_t len_;
  const std::string& host_;
  const std::string& port_;

  void connect_() {
    boost::asio::ip::tcp::resolver::query qr(host_, port_);
    boost::asio::connect(sk_, rs_.resolve(qr));
  }

  void write_(const char* data) {
    boost::asio::write(sk_, boost::asio::buffer(data, len_));
  }

  void read_() {
    char data[1024];
    len_ = boost::asio::read(sk_, boost::asio::buffer(data, len_));
    std::cout.write(data, len_);
    std::cout << '\n';
  }
};

template <typename T>
void doit() {
  boost::asio::io_service ios;
  T s([&ios]{
      short port = 12345;
      server s(ios, port);
      ios.run();
    });

  T c([&ios]{
      const std::string host = "localhost";
      const std::string port = "12345";
      const char* data = "abc";
      client c(ios, host, port);
      c.do_connect();
      c.do_write(data);
      c.do_read();
    });

  s.detach();

  c.join();
}

auto main() -> decltype(0)
{
  doit<std::thread>();
  return 0;
}
