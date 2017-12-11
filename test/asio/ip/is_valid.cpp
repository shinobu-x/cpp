#include <boost/asio/ip/address_v4.hpp>
#include <boost/asio/ip/address_v6.hpp>

#include <iostream>

auto main() -> decltype(0) {

  boost::system::error_code ec1, ec2;

  boost::asio::ip::address_v4 v4 =
    boost::asio::ip::address_v4::from_string("127.0.0.1", ec1);
  boost::asio::ip::address_v6 v6 =
    boost::asio::ip::address_v6::from_string("0::0", ec2);

  if (!ec1)
    std::cout << "fine\n";
  else
    std::cout << "bad\n";

  if (!ec2)
    std::cout << "fine\n";
  else
    std::cout << "bad\n";

  return 0;
}
