#include "../hpp/basic_raw_protocol.hpp"

#include <boost/asio/io_service.hpp>
#include <boost/asio/streambuf.hpp>
#include <boost/system/error_code.hpp>

#include <algorithm>
#include <ostream>

typedef basic_raw_protocol<AF_INET, AF_INET6, SOCK_RAW, IPPROTO_TCP> tcp_type;

auto main(int argc, char** argv) -> decltype(0)
{
  try {
    boost::asio::io_service ios;
    tcp_type::socket sk(ios, tcp_type::v4());

    tcp_type::resolver rs(ios);
    boost::asio::ip::address_v4::bytes_type b = {{127,0,0,1}};
    tcp_type::endpoint ep(boost::asio::ip::address_v4(b), 22);
    unsigned char buf[20];

    std::fill(buf, buf+sizeof(buf), 0xff);
    boost::asio::streambuf rq;
    std::ostream os(&rq);
    os.write(reinterpret_cast<char*>(buf), sizeof(buf));
    sk.send_to(rq.data(), ep);

  } catch (boost::system::error_code& ec) {
    return -1;
  }

  return 0;
}
