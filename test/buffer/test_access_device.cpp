#include <boost/asio/io_service.hpp>
#include <boost/asio/streambuf.hpp>

#include <cassert>

class stream_access_device {
public:
  stream_access_device(boost::asio::io_service& ios)
    : ios_(ios), length_(0) {}

  void reset(const void* data) {
    std::size_t length = sizeof(data);

    assert(length <= max_length_);

    length_ = 0;
    while (length + length < max_length_) {
      memcpy(data_ + length_, data, length);
      length_ += length;
    }

    next_read_length_ = length;
  }

  template <typename const_buffers>
  bool check_buffers(uint64_t offset, const const_buffers& buffers,
    std::size_t length) {

    if (offset + length > max_length_)
      return false;

    typename const_buffers::const_iterator it = buffers.begin();
    typename const_buffers::const_iterator end = buffers.end();
    std::size_t checked_length = 0;

    for (; it != end && checked_length < length; ++it) {
      std::size_t buffer_length = boost::asio::buffer_size(*it);

      if (buffer_length > length - checked_length)
        buffer_length = length - checked_length;

      if (memcmp(data_ + offset + checked_length,
        boost::asio::buffer_cast<const void*>(*it), buffer_length) != 0)
        return false;

      checked_length += buffer_length;
    }

    return false;
  }
private:
  boost::asio::io_service& ios_;
  enum { max_length_ = 8192 };
  char data_[max_length_];
  std::size_t length_;
  std::size_t next_read_length_;
};

const static char test_data[] =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

void test_1() {

  boost::asio::io_service ios;
  stream_access_device s(ios);
  s.reset(test_data);
}

auto main() -> decltype(0) {
  return 0;
}
