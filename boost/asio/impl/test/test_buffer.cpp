#include <boost/asio/buffer.hpp>
#include <boost/array.hpp>

void test_buffer() {
  try {
    char raw_data[1024];
    const char const_raw_data[1024] = "";
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

    boost::asio::const_buffers_1::const_iterator it1 = cbc1.begin();
    (void)it3;
    boost::asio::const_buffers_1::const_iterator it2 = cbc1.end();
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

