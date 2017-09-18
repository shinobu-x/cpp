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
    bi7 = boost::asio::buffers_iterator<
      boost::asio::const_buffers_1, char>::end(cb1);
    bi8 = boost::asio::buffers_iterator<
      boost::asio::const_buffers_1, const char>::end(cb1);
    bi9 = boost::asio::buffers_iterator<
      boost::array<boost::asio::const_buffer, 2>, char>::end(cb2);
    bi10 = boost::asio::buffers_iterator<
      boost::array<boost::asio::const_buffer, 2>, const char>::end(cb2);
    std_bi9 = boost::asio::buffers_iterator<
      std::array<boost::asio::const_buffer, 2>, char>::end(std_cb2);
    std_bi10 = boost::asio::buffers_iterator<
      std::array<boost::asio::const_buffer, 2>, const char>::end(std_cb2);
    bi11 = boost::asio::buffers_iterator<
      std::vector<boost::asio::const_buffer>, char>::end(cb3);
    bi12 = boost::asio::buffers_iterator<
      std::vector<boost::asio::const_buffer>, const char>::end(cb3);

    bi1 = boost::asio::buffers_begin(mb1);
    bi3 = boost::asio::buffers_begin(mb2);
    bi5 = boost::asio::buffers_begin(mb3);
    bi7 = boost::asio::buffers_begin(cb1);
    bi9 = boost::asio::buffers_begin(cb2);
    bi11 = boost::asio::buffers_begin(cb3);

    bi1 = boost::asio::buffers_begin(mb1);
    bi3 = boost::asio::buffers_begin(mb2);
    bi5 = boost::asio::buffers_begin(mb3);
    bi7 = boost::asio::buffers_begin(cb1);
    bi9 = boost::asio::buffers_begin(cb2);
    bi11 = boost::asio::buffers_begin(cb3);

    ++bi1; --bi1; bi1++; bi1--;
    --bi2; ++bi2; bi2--; bi2++;
    ++bi3; --bi3; bi3++; bi3--;
    --bi4; ++bi4; bi4--; bi4++;
    ++bi5; --bi5; bi5++; bi5--;
    --bi6; ++bi6; bi6--; bi6++;
    ++bi7; --bi7; bi7++; bi7--;
    --bi8; ++bi8; bi8--; bi8++;
    ++bi9; --bi9; bi9++; bi9--;
    --bi10; ++bi10; bi10--; bi10++;
    ++bi11; --bi11; bi11++; bi11--;
    --bi12; ++bi12; bi12--; bi12++;

    bi1 += 1; bi1 -= 1; bi1 = bi1 + 1; bi1 = bi1 - 1; bi1 = (+1) + bi1;
    bi2 -= 1; bi2 += 1; bi2 = bi2 - 1; bi2 = bi2 + 1; bi2 = (-1) + bi2;
    bi3 += 1; bi3 -= 1; bi3 = bi3 + 1; bi3 = bi3 - 1; bi3 = (+1) + bi3;
    bi4 -= 1; bi4 += 1; bi4 = bi4 - 1; bi4 = bi4 + 1; bi4 = (-1) + bi4;
    bi5 += 1; bi5 -= 1; bi5 = bi5 + 1; bi5 = bi5 - 1; bi5 = (+1) + bi5;
    bi6 -= 1; bi6 += 1; bi6 = bi6 - 1; bi6 = bi6 + 1; bi6 = (-1) + bi6;
    bi7 += 1; bi7 -= 1; bi7 = bi7 + 1; bi7 = bi7 - 1; bi7 = (+1) + bi7;
    bi8 -= 1; bi8 += 1; bi8 = bi8 - 1; bi8 = bi8 + 1; bi8 = (-1) + bi8;
    bi9 += 1; bi9 -= 1; bi9 = bi9 + 1; bi9 = bi9 - 1; bi9 = (+1) + bi9;
    bi10 -= 1; bi10 += 1; bi10 = bi10 - 1; bi10 = bi10 + 1; bi10 = (-1) + bi10;
    bi11 += 1; bi11 -= 1; bi11 = bi11 + 1; bi11 = bi11 - 1; bi11 = (+1) + bi11;
    bi12 -= 1; bi12 += 1; bi12 = bi12 - 1; bi12 = bi12 + 1; bi12 = (-1) + bi12;

    (void)static_cast<std::ptrdiff_t>(bi13 - bi1);
    (void)static_cast<std::ptrdiff_t>(bi14 - bi2);
    (void)static_cast<std::ptrdiff_t>(bi15 - bi3);
    (void)static_cast<std::ptrdiff_t>(bi16 - bi4);
    (void)static_cast<std::ptrdiff_t>(bi17 - bi5);
    (void)static_cast<std::ptrdiff_t>(bi18 - bi6);
    (void)static_cast<std::ptrdiff_t>(bi19 - bi7);
    (void)static_cast<std::ptrdiff_t>(bi20 - bi8);
    (void)static_cast<std::ptrdiff_t>(bi21 - bi9);
    (void)static_cast<std::ptrdiff_t>(bi22 - bi10);
    (void)static_cast<std::ptrdiff_t>(bi23 - bi11);
    (void)static_cast<std::ptrdiff_t>(bi24 - bi12);

  } catch (std::exception&) {}
}

auto main() -> decltype(0) {
  test_buffers_iterator();
  return 0;
}
