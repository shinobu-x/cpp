#include <boost/asio/buffers_iterator.hpp>
#include <boost/asio/buffer.hpp>

#include <boost/array.hpp>

#include <array>

void test_buffers_iterator() {
  try {
    char data1[16], data2[16];
    const char cdata1[16] = "", cdata2[16] = "";
    boost::asio::mutable_buffers_1 mb1 = boost::asio::buffer(data1);
    boost::array<boost::asio::mutable_buffer, 2> mb2 =
      {{boost::asio::buffer(data1), boost::asio::buffer(data2)}};
    std::array<boost::asio::mutable_buffer, 2> std_mb2 =
      {{boost::asio::buffer(data1), boost::asio::buffer(data2)}};
    std::vector<boost::asio::mutable_buffer> mb3;
    mb3.push_back(data1);
    boost::asio::const_buffers_1 cb1 = boost::asio::buffer(cdata1);
    boost::array<boost::asio::const_buffer, 2> cb2 =
      {{boost::asio::buffer(cdata1), boost::asio::buffer(cdata2)}};
    std::array<boost::asio::const_buffer, 2> std_cb2 =
      {{boost::asio::buffer(cdata1), boost::asio::buffer(cdata2)}};
    std::vector<boost::asio::const_buffer> cb3;
    cb3.push_back(boost::asio::buffer(cdata1));



  } catch (...) {}
}

auto main() -> decltype(0) {
  return 0;
} 
