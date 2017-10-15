#include <boost/asio/read_at.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/streambuf.hpp>

#include <boost/array.hpp>
#include <boost/bind.hpp>

#include <cassert>
#include <cstring>

#include "async_result.hpp"

class stream_access_device {
public:
  stream_access_device(boost::asio::io_service& ios)
    : ios_(ios), length_(0), next_read_length_(0) {}

  boost::asio::io_service& get_io_service() {
    return ios_;
  }

  void reset(const int* data, std::size_t length) {
    assert(length <= max_length_);

    length_ = 0;
    while (length_ + length < max_length_) {
      memcpy(data_ + length_, data, length);
      length_ = length;
    }

    next_read_length_ = length;
  }

  void next_read_length(std::size_t length) {
    next_read_length_ = length;
  }

  template <typename const_buffers>
  bool check_buffers(boost::asio::uint64_t offset, const const_buffers& buffers,
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
    return true;
  }

  template <typename mutable_buffers>
  std::size_t read_some_at(boost::asio::uint64_t offset,
    const mutable_buffers& buffers) {
    return boost::asio::buffer_copy(
      buffers, boost::asio::buffer(data_, length_) + offset, next_read_length_);
  }

  template <typename mutable_buffers>
  std::size_t read_some_at(boost::asio::uint64_t offset,
    const mutable_buffers& buffers, boost::system::error_code& ec) {
    ec = boost::system::error_code();
    return read_some_at(offset, buffers);
  }

  template <typename mutable_buffers, typename handler_type>
  void async_read_some_at(boost::asio::uint64_t offset,
    const mutable_buffers& buffers, handler_type handler) {
    std::size_t bytes_transferred = read_some_at(offset, buffers);
    ios_.post(boost::asio::detail::bind_handler(
      handler, boost::system::error_code(), bytes_transferred));
  }
private:
  boost::asio::io_service& ios_;
  enum { max_length_ = 8192 };
  char data_[max_length_];
  std::size_t length_;
  std::size_t next_read_length_;
};

static const char read_data[] =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

// Test mutable buffers_1 read_at
void test_1() {

}

auto main() -> decltype(0) {
  return 0;
}

