#include "../hpp/basic_raw_protocol.hpp"

#include <boost/array.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/streambuf.hpp>
#include <boost/system/error_code.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>

#include <cstdlib> /* size_t */

typedef basic_raw_protocol<AF_INET, AF_INET6, SOCK_RAW, IPPROTO_TCP> tcp_type;

void doit(char** argv) {
  boost::asio::io_service ios;
  tcp_type::socket sk(ios, tcp_type::v4());

  std::thread sender([&argv, &ios, &sk] {
    std::cout << std::this_thread::get_id() << '\n';
    tcp_type::resolver rs(ios);
    tcp_type::resolver::query q(tcp_type::v4(), argv[1], "");
    tcp_type::endpoint ep = *rs.resolve(q);
    boost::array<char, 1> buf = {{0}};
    sk.send_to(boost::asio::buffer(buf), ep);
  });

  std::thread receiver([&sk] {
    std::cout << std::this_thread::get_id() << '\n';
    boost::array<char, 128> buf;
    tcp_type::endpoint ep;
    size_t length = sk.receive_from(boost::asio::buffer(buf), ep);
    std::cout.write(buf.data(), length);
  });

  sender.join();
  receiver.join();
}
auto main(int argc, char** argv) -> decltype(0)
{
  doit(argv);
}
