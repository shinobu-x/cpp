#include <cstdlib>
#include <cstring>
#include <iostream>

#include <boost/asio.hpp>

auto main() -> decltype(0)
{
  boost::asio::io_service ios;
  boost::asio::ip::tcp::resolver rs(ios);
  boost::asio::ip::tcp::resolver::query q(boost::asio::ip::host_name(), "");
  boost::asio::ip::tcp::resolver::iterator it = rs.resolve(q);
  boost::asio::ip::tcp::resolver::iterator end;

  while (it != end) {
    boost::asio::ip::tcp::endpoint e = *it++;
    std::cout << e << '\n';
  }
/*
  boost::asio::ip::tcp::socket sk(ios);
  boost::asio::ip::tcp::resolver rs(ios);
  boost::asio::connect(sk, rs.resolve(
*/
  return 0;
}
