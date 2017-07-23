#include "../hpp/basic_raw_protocol.hpp"

#include <boost/array.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/streambuf.hpp>
#include <boost/system/error_code.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

#include <cstdlib> /* size_t */

typedef basic_raw_protocol<AF_INET, AF_INET6, SOCK_RAW, IPPROTO_TCP> tcp_type;

void doit(char** argv) {
  std::atomic<bool> is_ready{false};
  boost::asio::io_service ios;
  tcp_type::socket sk(ios, tcp_type::v4());

  std::thread sender([&argv, &is_ready, &ios, &sk] {
    try {
      std::cout << std::this_thread::get_id() << '\n';
      tcp_type::resolver rs(ios);
      tcp_type::resolver::query q(tcp_type::v4(), argv[1], "");
      tcp_type::endpoint ep = *rs.resolve(q);
      boost::array<char, 1> buf = {{0}};
      sk.send_to(boost::asio::buffer(buf), ep);
      std::this_thread::sleep_for(std::chrono::seconds(3));
      is_ready = true;
    } catch (boost::system::error_code& ec) {
      return -1;
    }
  });

  std::thread receiver([&is_ready, &sk] {
    while (!is_ready)
      std::this_thread::yield();

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
