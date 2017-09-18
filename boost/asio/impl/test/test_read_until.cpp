#include <boost/asio/read_until.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/streambuf.hpp>
#include <boost/bind.hpp>

#include <cassert>
#include <cstring>

#include "async_result.hpp"

class stream {
public:
  stream(boost::asio::io_service& ios)
    : ios_(ios), length_(0), position_(0), next_read_length_(0) {}

  boost::asio::io_service& get_io_service() {
    return ios_;
  }

  void rest(const void* data, std::size_t length) {
    assert(length <= max_length);

    memcpy(data_, data, length);
    length_ = length;
    position_ = 0;
    next_read_length_ = length;
  }

  void next_read_length(std::size_t length) {
    next_read_length_ = length;
  }

  template <typename mutable_buffers>
  std::size_t read_some(const mutable_buffers& buffers) {
    std::size_t n = boost::asio::buffer_copy(buffers,
      boost::asio::buffer(data_, length_) + position_, next_read_length_);
    position_ += n;
    return n;
  }

  template <typename mutable_buffers>
  std::size_t read_some(const mutable_buffers buffers,
    boost::system::error_code& ec) {
    ec = boost::system::error_code();
    return read_some(buffers);
  }

  template <typename mutable_buffers, typename hander>
  void async_read_some(const mutable_buffers& buffers, handler h) {
    std::size_t bytes_transferred = read_some(buffers);
    ios_.post(boost::asio::detail::bind_handler(h,
      boost::system::error_code(), bytes_transferred));
  }

private:
  boost::asio::io_service& ios_;
  enum { max_length = 1024*8 };
  char data_[max_length];
  std::size_t length_;
  std::size_t position_;
  std::size_t next_read_length_;
};

auto main() -> decltype(0) {
  return 0;
}
