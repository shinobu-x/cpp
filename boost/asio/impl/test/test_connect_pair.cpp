#include <boost/asio/local/connect_pair.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/local/datagram_protocol.hpp>
#include <boost/asio/local/stream_protocol.hpp>

// Test local connect pair
void test_1() {
  try {
    boost::asio::io_service ios;
    boost::system::error_code ec1;
    boost::system::error_code ec2;

    boost::asio::local::datagram_protocol::socket s1(ios);
    boost::asio::local::datagram_protocol::socket s2(ios);
    boost::asio::local::connect_pair(s1, s2);

    boost::asio::local::datagram_protocol::socket s3(ios);
    boost::asio::local::datagram_protocol::socket s4(ios);
    ec1 = boost::asio::local::connect_pair(s3, s4, ec2);

    boost::asio::local::stream_protocol::socket s5(ios);
    boost::asio::local::stream_protocol::socket s6(ios);
    boost::asio::local::connect_pair(s5, s6);

    boost::asio::local::stream_protocol::socket s7(ios);
    boost::asio::local::stream_protocol::socket s8(ios);
    ec1 = boost::asio::local::connect_pair(s7, s8, ec2);
  } catch (std::exception&) {}
}

auto main() -> decltype(0) {
  test_1();
  return 0;
}
