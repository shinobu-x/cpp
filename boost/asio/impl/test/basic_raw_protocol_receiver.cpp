#include "../hpp/basic_raw_protocol.hpp"

#include <boost/asio/io_service.hpp>
#include <boost/asio/buffer.hpp>
#include <boost/asio/placeholders.hpp>
#include <boost/system/error_code.hpp>
#include <boost/array.hpp>
#include <boost/bind.hpp>

#include <iostream>

typedef basic_raw_protocol<AF_INET, AF_INET6, SOCK_RAW, IPPROTO_TCP> tcp_type;

class receiver {
public:
  receiver(boost::asio::io_service& ios)
    : sk_(ios), ep1_(tcp_type::v4(), 0) {}

  bool open() {
    boost::system::error_code ec;
    if (!sk_.is_open())
      sk_.open(tcp_type::v4(), ec);

    return true;
  }

  bool bind() {
    boost::system::error_code ec;
    sk_.bind(ep1_, ec);
  }

  bool do_receive() {
   try {
/*
     sk_.async_receive_from(boost::asio::buffer(buf_), ep2_,
       boost::bind(&receiver::handler, this,
         boost::asio::placeholders::error,
         boost::asio::placeholders::bytes_transferred));
*/
      sk_.async_receive_from(boost::asio::buffer(buf_), ep2_,
        boost::bind(&receiver::handler, this,
          boost::asio::placeholders::error,
          boost::asio::placeholders::bytes_transferred));
    } catch (std::exception& e) {
      return false;
    }

    return true;
  }


private:
  void handler(boost::system::error_code ec, size_t bytes) {
    if (ec)
      if (ec != boost::asio::error::operation_aborted)
        return;
      else
        return;

    if (bytes != 0)
      std::cout << bytes << std::flush;
    do_receive();
  }

  tcp_type::socket sk_;
  tcp_type::endpoint ep1_;
  tcp_type::endpoint ep2_;
  boost::array<char, 2048> buf_;
};

auto main() -> decltype(0)
{
  boost::asio::io_service ios;
  receiver rv(ios);
  rv.open();
  rv.bind();
  rv.do_receive();
  ios.run();
  return 0;
}
