#include <boost/asio/buffer.hpp>
#include <boost/array.hpp>

#include <array>

void test_buffer() {
  try {
    char raw_data[1024];
    void* void_ptr_data = raw_data;
    const char const_raw_data[1024] = "";
    const void* const_void_ptr_data = const_raw_data;
    boost::array<char, 1024> array_data;
    const boost::array<char, 1024>& const_array_data_1 = array_data;
    boost::array<const char, 1024> const_array_data_2 = {{0}};
    std::array<char, 1024> std_array_data;
    const std::array<char, 1024>& const_std_array_data_1 = std_array_data;
    std::array<const char, 1024> const_std_array_data_2 = {{0}};
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

    boost::asio::mutable_buffers_1::const_iterator it1 = mbc1.begin();
    (void)it1;
    boost::asio::mutable_buffers_1::const_iterator it2 = mbc1.end();
    (void)it2;

    boost::asio::const_buffer cb1;
    boost::asio::const_buffer cb2(const_void_ptr_data, 1024);
    boost::asio::const_buffer cb3(cb1);
    boost::asio::const_buffer cb4(mb1);

    cb1 = cb2 + 128;
    cb1 = 128 + cb2;

    boost::asio::const_buffers_1 cbc1(cb1);
    boost::asio::const_buffers_1 cbc2(cbc1);

    boost::asio::const_buffers_1::const_iterator it3 = cbc1.begin();
    (void)it3;
    boost::asio::const_buffers_1::const_iterator it4 = cbc1.end();
    (void)it4;

    std::size_t s1 = boost::asio::buffer_size(mb1);
    (void)s1;
    std::size_t s2 = boost::asio::buffer_size(cb1);
    (void)s2;
    std::size_t s3 = boost::asio::buffer_size(mbc1);
    (void)s3;
    std::size_t s4 = boost::asio::buffer_size(cbc1);
    (void)s4;
    std::size_t s5 = boost::asio::buffer_size(mutable_buffer_sequence);
    (void)s5;
    std::size_t s6 = boost::asio::buffer_size(const_buffer_sequence);
    (void)s6;

    void* ptr1 = boost::asio::buffer_cast<void*>(mb1);
    (void)ptr1;
    const void* ptr2 = boost::asio::buffer_cast<const void*>(cb1);
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
    mb1 = boost::asio::buffer(std_array_data);
    mb1 = boost::asio::buffer(std_array_data, 1024);
    cb1 = boost::asio::buffer(const_std_array_data_1);
    cb1 = boost::asio::buffer(const_std_array_data_1, 1024);
    cb1 = boost::asio::buffer(const_std_array_data_2);
    cb1 = boost::asio::buffer(const_std_array_data_2, 1024);
    mb1 = boost::asio::buffer(vector_data);
    mb1 = boost::asio::buffer(vector_data, 1024);
    cb1 = boost::asio::buffer(const_vector_data);
    cb1 = boost::asio::buffer(const_vector_data, 1024);
    cb1 = boost::asio::buffer(string_data);
    cb1 = boost::asio::buffer(string_data, 1024);

    std::size_t s7 = boost::asio::buffer_copy(mb1, cb2);
    (void)s7;
    std::size_t s8 = boost::asio::buffer_copy(mb1, cb2);
    (void)s8;
    std::size_t s9 = boost::asio::buffer_copy(mb1, mb2);
    (void)s9;
    std::size_t s10 = boost::asio::buffer_copy(mb1, mbc2);
    (void)s10;
    std::size_t s11 = boost::asio::buffer_copy(mb1, const_buffer_sequence);
    (void)s11;
    std::size_t s12 = boost::asio::buffer_copy(mbc1, cb2);
    (void)s12;
    std::size_t s13 = boost::asio::buffer_copy(mbc1, cbc2);
    (void)s13;
    std::size_t s14 = boost::asio::buffer_copy(mbc1, mb2);
    (void)s14;
    std::size_t s15 = boost::asio::buffer_copy(mbc1, mbc2);
    (void)s15;
    std::size_t s16 = boost::asio::buffer_copy(mbc1, const_buffer_sequence);
    (void)s16;
    std::size_t s17 = boost::asio::buffer_copy(mutable_buffer_sequence, cb2);
    (void)s17;
    std::size_t s18 = boost::asio::buffer_copy(mutable_buffer_sequence, cbc2);
    (void)s18;
    std::size_t s19 = boost::asio::buffer_copy(mutable_buffer_sequence, mb2);
    (void)s19;
    std::size_t s20 = boost::asio::buffer_copy(mutable_buffer_sequence, mbc2);
    (void)s20;
    std::size_t s21 =
      boost::asio::buffer_copy(mutable_buffer_sequence, const_buffer_sequence);
    (void)s21;
    std::size_t s22 = boost::asio::buffer_copy(mb1, cb2, 128);
    (void)s22;
    std::size_t s23 = boost::asio::buffer_copy(mb1, cbc2, 128);
    (void)s23;
    std::size_t s24 = boost::asio::buffer_copy(mb1, mb2, 128);
    (void)s24;
    std::size_t s25 = boost::asio::buffer_copy(mb1, mbc2, 128);
    (void)s25;
    std::size_t s26 = boost::asio::buffer_copy(mb1, const_buffer_sequence, 128);
    (void)s26;
    std::size_t s27 = boost::asio::buffer_copy(mbc1, cb2, 128);
    (void)s27;
    std::size_t s28 = boost::asio::buffer_copy(mbc1, cbc2, 128);
    (void)s28;
    std::size_t s29 = boost::asio::buffer_copy(mbc1, mbc2, 128);
    (void)s29;
    std::size_t s30 = boost::asio::buffer_copy(mbc1, mbc2, 128);
    (void)s30;
    std::size_t s31 =
      boost::asio::buffer_copy(mbc1, const_buffer_sequence, 128);
    (void)s31;
    std::size_t s32 =
      boost::asio::buffer_copy(mutable_buffer_sequence, cb2, 128);
    (void)s32;
    std::size_t s33 =
      boost::asio::buffer_copy(mutable_buffer_sequence, cbc2, 128);
    (void)s33;
    std::size_t s34 =
      boost::asio::buffer_copy(mutable_buffer_sequence, mb2, 128);
    (void)s34;
    std::size_t s35 =
      boost::asio::buffer_copy(mutable_buffer_sequence, mbc2, 128);
    (void)s35;
    std::size_t s36 =
      boost::asio::buffer_copy(
        mutable_buffer_sequence, const_buffer_sequence, 128);
    (void)s36;
  } catch (...) {}
}

auto main() -> decltype(0) {
  test_buffer();
  return 0;
}
