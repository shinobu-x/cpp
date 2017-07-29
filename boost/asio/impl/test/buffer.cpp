#include <boost/asio/buffer.hpp>
#include <boost/array.hpp>
#include <boost/system/error_code.hpp>

auto main() -> decltype(0) {
try {
  char raw_data[1024];
  const char const_raw_data[1024] = "";
  void* void_ptr_data = raw_data;
  const void* const_void_ptr_data = const_raw_data;

  boost::array<char, 1024> array_data;
  const boost::array<char, 1024>& const_array_data_1 = array_data;
  boost::array<const char, 1024> const_array_data_2 = {{0}};

  std::vector<char> vector_data(1024);
  const std::vector<char>& const_vector_data = vector_data;
  const std::string string_data(1024, ' ');
  std::vector<boost::asio::mutable_buffer> mutable_buffer_sequence;
  std::vector<boost::asio::const_buffer> const_buffer_sequence;

  boost::asio::mutable_buffer mb1;
  boost::asio::mutable_buffer mb2(void_ptr_data, 1024);
  boost::asio::mutable_buffer mb3(mb1);

  mb1 = mb2 + 128;
  mb1 = 128 + mb2;

  boost::asio::mutable_buffers_1 mbc1(mb1);
  boost::asio::mutable_buffers_1 mbc2(mbc1);

  boost::asio::mutable_buffers_1::const_iterator iter1 = mbc1.begin();
  boost::asio::mutable_buffers_1::const_iterator iter2 = mbc1.end();

  (void)iter1;
  (void)iter2;

  boost::asio::const_buffer cb1;
  boost::asio::const_buffer cb2(const_void_ptr_data, 1024);
  boost::asio::const_buffer cb3(cb1);
  boost::asio::const_buffer cb4(mb1);

  cb1 = cb2 + 128;
  cb2 = 128 + cb1;

  boost::asio::const_buffers_1 cbc1(cb1);
  boost::asio::const_buffers_1 cbc2(cbc1);

  boost::asio::const_buffers_1::const_iterator iter3 = cbc1.begin();
  boost::asio::const_buffers_1::const_iterator iter4 = cbc1.end();

  std::size_t size1 = boost::asio::buffer_size(mb1);
  std::size_t size2 = boost::asio::buffer_size(cb1);
  std::size_t size3 = boost::asio::buffer_size(mbc1);
  std::size_t size4 = boost::asio::buffer_size(cbc1);
  std::size_t size5 = boost::asio::buffer_size(mutable_buffer_sequence);
  std::size_t size6 = boost::asio::buffer_size(const_buffer_sequence);

  (void)size1;
  (void)size2;
  (void)size3;
  (void)size4;
  (void)size5;
  (void)size6;

  void* ptr1 = boost::asio::buffer_cast<void*>(mb1);
  const void* ptr2 = boost::asio::buffer_cast<const void*>(cb1);

  (void)ptr1;
  (void)ptr2;

  mb1 = boost::asio::buffer(mb2);
  mb1 = boost::asio::buffer(mb2, 128);
  cb1 = boost::asio::buffer(cb2);
  cb1 = boost::asio::buffer(cb2, 128);
  mb1 = boost::asio::buffer(void_ptr_data, 1024);
  cb1 = boost::asio::buffer(const_void_ptr_data, 1024);
  mb1 = boost::asio::buffer(raw_data);
  mb1 = boost::asio::buffer(raw_data, 1024);
  cb1 = boost::asio::buffer(const_raw_data);
  cb1 = boost::asio::buffer(const_raw_data, 1024);
  mb1 = boost::asio::buffer(array_data);
  mb1 = boost::asio::buffer(array_data, 1024);
  cb1 = boost::asio::buffer(const_array_data_1);
  cb1 = boost::asio::buffer(const_array_data_1, 1024);
  cb1 = boost::asio::buffer(const_array_data_2);
  cb1 = boost::asio::buffer(const_array_data_2, 1024);
  mb1 = boost::asio::buffer(vector_data);
  mb1 = boost::asio::buffer(vector_data, 1024);
  cb1 = boost::asio::buffer(const_vector_data);
  cb1 = boost::asio::buffer(const_vector_data, 1024);
  cb1 = boost::asio::buffer(string_data);
  cb1 = boost::asio::buffer(string_data, 1024);

  std::size_t size7 = boost::asio::buffer_copy(mb1, cb2);
  std::size_t size8 = boost::asio::buffer_copy(mb1, cbc2);
  std::size_t size9 = boost::asio::buffer_copy(mb1, mb2);
  std::size_t size10 = boost::asio::buffer_copy(mb1, mbc2);
  std::size_t size11 = boost::asio::buffer_copy(mb1, const_buffer_sequence);
  std::size_t size12 = boost::asio::buffer_copy(mbc1, cb2);
  std::size_t size13 = boost::asio::buffer_copy(mbc1, cbc2);
  std::size_t size14 = boost::asio::buffer_copy(mbc1, mb2);
  std::size_t size15 = boost::asio::buffer_copy(mbc1, mbc2);
  std::size_t size16 = boost::asio::buffer_copy(mbc1, mbc2);
  std::size_t size17 = boost::asio::buffer_copy(mutable_buffer_sequence, cb2);
  std::size_t size18 = boost::asio::buffer_copy(mutable_buffer_sequence, cbc2);
  std::size_t size19 = boost::asio::buffer_copy(mutable_buffer_sequence, mb2);
  std::size_t size20 = boost::asio::buffer_copy(mutable_buffer_sequence, mbc2);
  std::size_t size21 = boost::asio::buffer_copy(
    mutable_buffer_sequence, const_buffer_sequence);
  std::size_t size22 = boost::asio::buffer_copy(mb1, cb2, 128);
  std::size_t size23 = boost::asio::buffer_copy(mb1, cbc2, 128);
  std::size_t size24 = boost::asio::buffer_copy(mb1, mb2, 128);
  std::size_t size25 = boost::asio::buffer_copy(mb1, mbc2, 128);
  std::size_t size26 = boost::asio::buffer_copy(
    mb1, const_buffer_sequence, 128);
  std::size_t size27 = boost::asio::buffer_copy(mbc1, cb2, 128);
  std::size_t size28 = boost::asio::buffer_copy(mbc1, cbc2, 128);
  std::size_t size29 = boost::asio::buffer_copy(mbc1, mb2, 128);
  std::size_t size30 = boost::asio::buffer_copy(mbc1, mbc2, 128);
  std::size_t size31 = boost::asio::buffer_copy(
    mbc1, const_buffer_sequence, 128);
  std::size_t size32 = boost::asio::buffer_copy(
    mutable_buffer_sequence, cb2, 128);
  std::size_t size33 = boost::asio::buffer_copy(
    mutable_buffer_sequence, cbc2, 128);
  std::size_t size34 = boost::asio::buffer_copy(
    mutable_buffer_sequence, mb2, 128);
  std::size_t size35 = boost::asio::buffer_copy(
    mutable_buffer_sequence, mbc2, 128);
  std::size_t size36 = boost::asio::buffer_copy(
    mutable_buffer_sequence, const_buffer_sequence, 128);

  (void)size7;
  (void)size8;
  (void)size9;
  (void)size10;
  (void)size11;
  (void)size12;
  (void)size13;
  (void)size14;
  (void)size15;
  (void)size16;
  (void)size17;
  (void)size18;
  (void)size19;
  (void)size20;
  (void)size21;
  (void)size22;
  (void)size23;
  (void)size24;
  (void)size25;
  (void)size26;
  (void)size27;
  (void)size28;
  (void)size29;
  (void)size30;
  (void)size31;
  (void)size32;
  (void)size33;
  (void)size34;
  (void)size35;
  (void)size36;
}catch (boost::system::error_code& ec) {
}
} // main
