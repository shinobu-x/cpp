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
    mb3.push_back(boost::asio::buffer(data1));
    boost::asio::const_buffers_1 cb1 = boost::asio::buffer(cdata1);
    boost::array<boost::asio::const_buffer, 2> cb2 =
      {{boost::asio::buffer(cdata1), boost::asio::buffer(cdata2)}};
    std::array<boost::asio::const_buffer, 2> std_cb2 =
      {{boost::asio::buffer(cdata1), boost::asio::buffer(cdata2)}};
    std::vector<boost::asio::const_buffer> cb3;
    cb3.push_back(boost::asio::buffer(cdata1));

    boost::asio::buffers_iterator<boost::asio::mutable_buffers_1, char> bi1;
    boost::asio::buffers_iterator<
      boost::asio::mutable_buffers_1, const char> bi2;
    boost::asio::buffers_iterator<
      boost::array<boost::asio::mutable_buffer, 2>, char> bi3;
    boost::asio::buffers_iterator<
      boost::array<boost::asio::mutable_buffer, 2>, const char> bi4;
    boost::asio::buffers_iterator<
      std::array<boost::asio::mutable_buffer, 2>, char> std_bi3;
    boost::asio::buffers_iterator<
      std::array<boost::asio::mutable_buffer, 2>, const char> std_bi4;
    boost::asio::buffers_iterator<
      std::vector<boost::asio::mutable_buffer>, char> bi5;
    boost::asio::buffers_iterator<
      std::vector<boost::asio::mutable_buffer>, const char> bi6;
    boost::asio::buffers_iterator<boost::asio::const_buffers_1, char> bi7;
    boost::asio::buffers_iterator<boost::asio::const_buffers_1, const char> bi8;
    boost::asio::buffers_iterator<
      boost::array<boost::asio::const_buffer, 2>, char> bi9;
    boost::asio::buffers_iterator<
      boost::array<boost::asio::const_buffer, 2>, const char> bi10;
    boost::asio::buffers_iterator<
      std::array<boost::asio::const_buffer, 2>, char> std_bi9;
    boost::asio::buffers_iterator<
      std::array<boost::asio::const_buffer, 2>, const char> std_bi10;
    boost::asio::buffers_iterator<
      std::vector<boost::asio::const_buffer>, char> bi11;
    boost::asio::buffers_iterator<
      std::vector<boost::asio::const_buffer>, const char> bi12;

    boost::asio::buffers_iterator<
      boost::asio::mutable_buffers_1, char> bi13(
        boost::asio::buffers_iterator<
          boost::asio::mutable_buffers_1, char>::begin(mb1));
    boost::asio::buffers_iterator<
      boost::asio::mutable_buffers_1, const char> bi14(
        boost::asio::buffers_iterator<
          boost::asio::mutable_buffers_1, const char>::begin(mb1));
    boost::asio::buffers_iterator<
      boost::array<boost::asio::mutable_buffer, 2>, char> bi15(
        boost::asio::buffers_iterator<
          boost::array<boost::asio::mutable_buffer, 2>, char>::begin(mb2));
    boost::asio::buffers_iterator<
      boost::array<boost::asio::mutable_buffer, 2>, const char> bi16(
        boost::asio::buffers_iterator<
          boost::array<
            boost::asio::mutable_buffer, 2>, const char>::begin(mb2));
    boost::asio::buffers_iterator<
      std::array<boost::asio::mutable_buffer, 2>, char> std_bi15(
        boost::asio::buffers_iterator<
          std::array<boost::asio::mutable_buffer, 2>, char>::begin(std_mb2));
    boost::asio::buffers_iterator<
      std::array<boost::asio::mutable_buffer, 2>, const char> std_bi16(
        boost::asio::buffers_iterator<
          std::array<
            boost::asio::mutable_buffer, 2>, const char>::begin(std_mb2)); 
    boost::asio::buffers_iterator<
      std::vector<boost::asio::mutable_buffer>, char> bi17(
        boost::asio::buffers_iterator<
          std::vector<boost::asio::mutable_buffer>, char>::begin(mb3));
    boost::asio::buffers_iterator<
      std::vector<boost::asio::mutable_buffer>, const char> bi18(
        boost::asio::buffers_iterator<
          std::vector<boost::asio::mutable_buffer>, const char>::begin(mb3));
    boost::asio::buffers_iterator<
      boost::asio::const_buffers_1, char> bi19(
        boost::asio::buffers_iterator<
          boost::asio::const_buffers_1, char>::begin(cb1));
    boost::asio::buffers_iterator<
      boost::asio::const_buffers_1, const char> bi20(
        boost::asio::buffers_iterator<
          boost::asio::const_buffers_1, const char>::begin(cb1));
    boost::asio::buffers_iterator<
      boost::array<boost::asio::const_buffer, 2>, char> bi21(
        boost::asio::buffers_iterator<
          boost::array<boost::asio::const_buffer, 2>, char>::begin(cb2));
    boost::asio::buffers_iterator<
      boost::array<boost::asio::const_buffer, 2>, const char> bi22(
        boost::asio::buffers_iterator<
          boost::array<boost::asio::const_buffer, 2>, const char>::begin(cb2));
    boost::asio::buffers_iterator<
      std::array<boost::asio::const_buffer, 2>, char> std_bi21(
        boost::asio::buffers_iterator<
          std::array<boost::asio::const_buffer, 2>, char>::begin(std_cb2));
    boost::asio::buffers_iterator<
      std::array<boost::asio::const_buffer, 2>, const char> std_bi22(
        boost::asio::buffers_iterator<
          std::array<boost::asio::const_buffer, 2>,
            const char>::begin(std_cb2));
    boost::asio::buffers_iterator<
      std::vector<boost::asio::const_buffer>, char> bi23(
        boost::asio::buffers_iterator<
          std::vector<boost::asio::const_buffer>, char>::begin(cb3));
    boost::asio::buffers_iterator<
      std::vector<boost::asio::const_buffer>, const char> bi24(
        boost::asio::buffers_iterator<
          std::vector<boost::asio::const_buffer>, const char>::begin(cb3));

    bi1 = boost::asio::buffers_iterator<
      boost::asio::mutable_buffers_1, char>::begin(mb1);
    bi2 = boost::asio::buffers_iterator<
      boost::asio::mutable_buffers_1, const char>::begin(mb1);
    bi3 = boost::asio::buffers_iterator<
      boost::array<boost::asio::mutable_buffer, 2>, char>::begin(mb2);
    bi4 = boost::asio::buffers_iterator<
      boost::array<boost::asio::mutable_buffer, 2>, const char>::begin(mb2);
    std_bi3 = boost::asio::buffers_iterator<
      std::array<boost::asio::mutable_buffer, 2>, char>::begin(std_mb2);
    std_bi4 = boost::asio::buffers_iterator<
      std::array<boost::asio::mutable_buffer, 2>, const char>::begin(std_mb2);
    bi5 = boost::asio::buffers_iterator<
      std::vector<boost::asio::mutable_buffer>, char>::begin(mb3);
    bi6 = boost::asio::buffers_iterator<
      std::vector<boost::asio::mutable_buffer>, const char>::begin(mb3);
    bi7 = boost::asio::buffers_iterator<
      boost::asio::const_buffers_1, char>::begin(cb1);
    bi8 = boost::asio::buffers_iterator<
      boost::asio::const_buffers_1, const char>::begin(cb1);
    bi9 = boost::asio::buffers_iterator<
      boost::array<boost::asio::const_buffer, 2>, char>::begin(cb2);
    bi10 = boost::asio::buffers_iterator<
      boost::array<boost::asio::const_buffer, 2>, const char>::begin(cb2);
    std_bi9 = boost::asio::buffers_iterator<
      std::array<boost::asio::const_buffer, 2>, char>::begin(std_cb2);
    std_bi10 = boost::asio::buffers_iterator<
      std::array<boost::asio::const_buffer, 2>, const char>::begin(std_cb2);
    bi11 = boost::asio::buffers_iterator<
      std::vector<boost::asio::const_buffer>, char>::begin(cb3);
    bi12 = boost::asio::buffers_iterator<
      std::vector<boost::asio::const_buffer>, const char>::begin(cb3);

    bi1 = boost::asio::buffers_iterator<
      boost::asio::mutable_buffers_1, char>::end(mb1);
    bi2 = boost::asio::buffers_iterator<
      boost::asio::mutable_buffers_1, const char>::end(mb1);
    bi3 = boost::asio::buffers_iterator<
      boost::array<boost::asio::mutable_buffer, 2>, char>::end(mb2);
    bi4 = boost::asio::buffers_iterator<
      boost::array<boost::asio::mutable_buffer, 2>, const char>::end(mb2);
    std_bi3 = boost::asio::buffers_iterator<
      std::array<boost::asio::mutable_buffer, 2>, char>::end(std_mb2);
    std_bi4 = boost::asio::buffers_iterator<
      std::array<boost::asio::mutable_buffer, 2>, const char>::end(std_mb2);
    bi5 = boost::asio::buffers_iterator<
      std::vector<boost::asio::mutable_buffer>, char>::end(mb3);
    bi6 = boost::asio::buffers_iterator<
      std::vector<boost::asio::mutable_buffer>, const char>::end(mb3);
  } catch (...) {}
}

auto main() -> decltype(0) {
  return 0;
} 
