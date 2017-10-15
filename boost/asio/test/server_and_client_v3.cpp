#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <utility>

#include <cstdlib> /** size_t **/

#include <boost/asio.hpp>

#define M_MAX_LENGTH(a)     ((1024) * (a))
#define M_SIZE              4

class session
  : public std::enable_shared_from_this<session> {
private:
  boost::asio::ip::tcp::socket sk_;
  char data_[M_MAX_LENGTH(M_SIZE)];

public:
  session(boost::asio::ip::tcp::socket sk)
    : sk_(std::move(sk)) {
    std::cout << "Connected..." << '\n';
  }

  void do_read() {
    read_();
  }

private:
  void read_() {
    std::cout << "Reading..." << '\n';
    // For lambda expression
    auto self(shared_from_this());
    sk_.async_read_some(
      boost::asio::buffer(data_, M_MAX_LENGTH(M_SIZE)),
      [this, self](boost::system::error_code ec, std::size_t length) {
        if (!ec)
          write_(length);
      }
    );
  }

  void write_(size_t length) {
    std::cout << "Writing..." << '\n';
    // For lambda expression
    auto self(shared_from_this());
    boost::asio::async_write(
      sk_,
      boost::asio::buffer(data_, length),
      [this, self](boost::system::error_code ec, std::size_t) {
        if (!ec)
          read_();
      }
    ); 
  }
};  // End session

class server {
private:
  boost::asio::ip::tcp::acceptor ap_;
  boost::asio::ip::tcp::socket sk_;

public:
  server(boost::asio::io_service& ios, unsigned short port)
    : ap_(ios,
        boost::asio::ip::tcp::endpoint(
          boost::asio::ip::tcp::v4(),
          port)),
      sk_(ios) {
    std::cout << "Listening..." << '\n';
    accept_();
  }

private:
  void accept_() {
    ap_.async_accept(sk_,
      [this](boost::system::error_code ec) {
        if (!ec)
          // Start session and trigger read op
          std::make_shared<session>(std::move(sk_))->do_read();
          accept_();
      }
    );
  }
};  // End server

class client {
private:
  boost::asio::ip::tcp::socket sk_;
  boost::asio::ip::tcp::resolver rs_;
  size_t len_;
  const std::string& host_;
  const std::string& port_;

public:
  client(boost::asio::io_service& ios,
    const std::string& host, const std::string& port)
    : rs_(ios), sk_(ios), host_(host), port_(port) {}
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
  void connect_() {
    boost::asio::ip::tcp::resolver::query qr(host_, port_);
    boost::asio::connect(sk_, rs_.resolve(qr));
  }

  void write_(const char* data) {
    boost::asio::write(sk_, boost::asio::buffer(data, len_));
  }

  void read_() {
    char data[M_MAX_LENGTH(M_SIZE)];
    len_ = boost::asio::read(sk_, boost::asio::buffer(data, len_));
    std::cout.write(data, len_);
    std::cout << '\n';
  }
};  // End client

template <typename T>
void doit() {
  boost::asio::io_service ios;
  T s([&ios] {
    short port = 12345;
    server s(ios, port);
    ios.run();
  });

  T c([&ios] {
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
}
