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

  void reset(const void* data, std::size_t length) {
    assert(length <= max_length_);

    length_ = 0;
    while (length_ + length < max_length_) {
      memcpy(data_ + length_, data, length);
      length_ += length;
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
  boost::asio::io_service ios;
  stream_access_device s(ios);
  char read_buf[sizeof(read_data)];
  boost::asio::mutable_buffers_1 buffers =
    boost::asio::buffer(read_buf, sizeof(read_buf));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  std::size_t bytes_transferred = boost::asio::read_at(s, 0, buffers);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1, buffers);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));
}

// Test std vector buffers read_at
void test_2() {
  boost::asio::io_service ios;
  stream_access_device s(ios);
  char read_buf[sizeof(read_data)];
  std::vector<boost::asio::mutable_buffer> buffers;
  buffers.push_back(boost::asio::buffer(read_buf, 32));
  buffers.push_back(boost::asio::buffer(read_buf) + 32);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  std::size_t bytes_transferred = boost::asio::read_at(s, 0, buffers);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));
}

void test_3() {
  boost::asio::io_service ios;
  stream_access_device s(ios);
  boost::asio::streambuf sb(sizeof(read_data));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  std::size_t bytes_transferred = boost::asio::read_at(s, 0, sb);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(1234, sb.data(), sizeof(read_data)));
}

// Test nothrow mutable_buffers_1 read_at
void test_4() {
  boost::asio::io_service ios;
  stream_access_device s(ios);
  char read_buf[sizeof(read_data)];
  boost::asio::mutable_buffers_1 buffers =
    boost::asio::buffer(read_buf, sizeof(read_buf));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  boost::system::error_code ec;
  std::size_t bytes_transferred = boost::asio::read_at(s, 0, buffers, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));
}

// Test nothrow vector buffers read_at
void test_5() {
  boost::asio::io_service ios;
  stream_access_device s(ios);
  char read_buf[sizeof(read_data)];
  std::vector<boost::asio::mutable_buffer> buffers;
  buffers.push_back(boost::asio::buffer(read_buf, 32));
  buffers.push_back(boost::asio::buffer(read_buf) + 32);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  boost::system::error_code ec;
  std::size_t bytes_transferred = boost::asio::read_at(s, 0, buffers, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));
}

// Test nothrow streambuf read_at
void test_6() {
  boost::asio::io_service ios;
  stream_access_device s(ios);
  boost::asio::streambuf sb(sizeof(read_data));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  boost::system::error_code ec;
  std::size_t bytes_transferred = boost::asio::read_at(s, 0, sb, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(1234, sb.data(), sizeof(read_data)));
}

bool old_style_transfer_all(const boost::system::error_code& ec, std::size_t) {
  return !!ec;
}

std::size_t short_transfer(const boost::system::error_code& ec, std::size_t) {
  return !!ec ? 0 : 3;
}

// Test mutable buffers_1 read_at
void test_7() {
  boost::asio::io_service ios;
  stream_access_device s(ios);
  char read_buf[sizeof(read_data)];
  boost::asio::mutable_buffers_1 buffers =
    boost::asio::buffer(read_buf, sizeof(read_buf));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  std::size_t bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_all());
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_all());
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_all());
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_all());
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_all());
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_all());
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(1));
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(1));
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0 ,sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(1));
  assert(bytes_transferred == 1);
  assert(s.check_buffers(0, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(1));
  assert(bytes_transferred == 1);
  assert(s.check_buffers(1234, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(1));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(1));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(42));
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(42));
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(42));
  assert(bytes_transferred == 42);
  assert(s.check_buffers(0, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(42));
  assert(bytes_transferred == 42);
  assert(s.check_buffers(1234, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(42));
  assert(bytes_transferred == 50);
  assert(s.check_buffers(0, buffers, 50));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(42));
  assert(bytes_transferred == 50);
  assert(s.check_buffers(1234, buffers, 50));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(1));
  assert(bytes_transferred == 1);
  assert(s.check_buffers(0, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(1));
  assert(bytes_transferred == 1);
  assert(s.check_buffers(1234, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(1));
  assert(bytes_transferred == 1);
  assert(s.check_buffers(0, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(1));
  assert(bytes_transferred == 1);
  assert(s.check_buffers(1234, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(1));
  assert(bytes_transferred == 1);
  assert(s.check_buffers(0, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(1));
  assert(bytes_transferred == 1);
  assert(s.check_buffers(1234, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(10));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(10));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(10));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(10));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(10));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(10));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(42));
  assert(bytes_transferred == 42);
  assert(s.check_buffers(0, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(42));
  assert(bytes_transferred == 42);
  assert(s.check_buffers(1234, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(42));
  assert(bytes_transferred == 42);
  assert(s.check_buffers(0, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(42));
  assert(bytes_transferred == 42);
  assert(s.check_buffers(1234, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(42));
  assert(bytes_transferred == 42);
  assert(s.check_buffers(0, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(42));
  assert(bytes_transferred == 42);
  assert(s.check_buffers(1234, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    old_style_transfer_all);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    old_style_transfer_all);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    old_style_transfer_all);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    old_style_transfer_all);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    old_style_transfer_all);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    old_style_transfer_all);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers, short_transfer);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers, short_transfer);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers, short_transfer);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers, short_transfer);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers, short_transfer);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers, short_transfer);
  assert(bytes_transferred = sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));
}

// Test vector buffers read_at
void test_8() {
  boost::asio::io_service ios;
  stream_access_device s(ios);
  char read_buf[sizeof(read_data)];
  std::vector<boost::asio::mutable_buffer> buffers;
  buffers.push_back(boost::asio::buffer(read_buf, 32));
  buffers.push_back(boost::asio::buffer(read_buf) + 32);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  std::size_t bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_all());
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_all());
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_all());
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_all());
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_all());
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_all());
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(1));
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(1));
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(1));
  assert(bytes_transferred == 1);
  assert(s.check_buffers(0, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(1));
  assert(bytes_transferred == 1);
  assert(s.check_buffers(1234, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(1));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(1));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(42));
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(42));
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(42));
  assert(bytes_transferred == 42);
  assert(s.check_buffers(0, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(42));
  assert(bytes_transferred == 42);
  assert(s.check_buffers(1234, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(42));
  assert(bytes_transferred == 50);
  assert(s.check_buffers(0, buffers, 50));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(42));
  assert(bytes_transferred == 50);
  assert(s.check_buffers(1234, buffers, 50));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(1));
  assert(bytes_transferred == 1);
  assert(s.check_buffers(0, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(1));
  assert(bytes_transferred == 1);
  assert(s.check_buffers(1234, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(1));
  assert(bytes_transferred == 1);
  assert(s.check_buffers(0, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(1));
  assert(bytes_transferred == 1);
  assert(s.check_buffers(1234, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(1));
  assert(bytes_transferred == 1);
  assert(s.check_buffers(0, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(1));
  assert(bytes_transferred == 1);
  assert(s.check_buffers(1234, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(10));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(10));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(10));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(10));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(10));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(10));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(42));
  assert(bytes_transferred == 42);
  assert(s.check_buffers(0, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(42));
  assert(bytes_transferred == 42);
  assert(s.check_buffers(1234, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(42));
  assert(bytes_transferred == 42);
  assert(s.check_buffers(0, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(42));
  assert(bytes_transferred == 42);
  assert(s.check_buffers(1234, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(42));
  assert(bytes_transferred == 42);
  assert(s.check_buffers(0, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(42));
  assert(bytes_transferred == 42);
  assert(s.check_buffers(1234, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    old_style_transfer_all);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    old_style_transfer_all);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    old_style_transfer_all);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    old_style_transfer_all);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    old_style_transfer_all);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    old_style_transfer_all);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers, short_transfer);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers, short_transfer);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers, short_transfer);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers, short_transfer);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers, short_transfer);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers, short_transfer);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));
}

// Test streambuf read_at
void test_9() {
  boost::asio::io_service ios;
  stream_access_device s(ios);
  boost::asio::streambuf sb(sizeof(read_data));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  std::size_t bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_all());
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_all());
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_all());
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_all());
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_all());
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_all());
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_at_least(1));
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_at_least(1));
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_at_least(1));
  assert(bytes_transferred == 1);
  assert(sb.size() == 1);
  assert(s.check_buffers(0, sb.data(), 1));
 
  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_at_least(1));
  assert(bytes_transferred == 1);
  assert(sb.size() == 1);
  assert(s.check_buffers(1234, sb.data(), 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_at_least(1));
  assert(bytes_transferred == 10);
  assert(sb.size() == 10);
  assert(s.check_buffers(0, sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_at_least(1));
  assert(bytes_transferred == 10);
  assert(sb.size() == 10);
  assert(s.check_buffers(1234, sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == 10);
  assert(sb.size() == 10);
  assert(s.check_buffers(0, sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == 10);
  assert(sb.size() == 10);
  assert(s.check_buffers(1234, sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == 10);
  assert(sb.size() == 10);
  assert(s.check_buffers(0, sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == 10);
  assert(sb.size() == 10);
  assert(s.check_buffers(1234, sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(50);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == 50);
  assert(sb.size() == 50);
  assert(s.check_buffers(0, sb.data(), 50));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(50);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == 50);
  assert(sb.size() == 50);
  assert(s.check_buffers(1234, sb.data(), 50));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_at_least(42));
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_at_least(42));
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_at_least(42));
  assert(bytes_transferred == 42);
  assert(sb.size() == 42);
  assert(s.check_buffers(0, sb.data(), 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_at_least(42));
  assert(bytes_transferred == 42);
  assert(sb.size() == 42);
  assert(s.check_buffers(1234, sb.data(), 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_at_least(42));
  assert(bytes_transferred == 50);
  assert(sb.size() == 50);
  assert(s.check_buffers(0, sb.data(), 50));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_at_least(42));
  assert(bytes_transferred == 50);
  assert(sb.size() == 50);
  assert(s.check_buffers(1234, sb.data(), 50));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_exactly(1));
  assert(bytes_transferred == 1);
  assert(sb.size() == 1);
  assert(s.check_buffers(0, sb.data(), 1));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_exactly(1));
  assert(bytes_transferred == 1);
  assert(sb.size() == 1);
  assert(s.check_buffers(1234, sb.data(), 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_exactly(1));
  assert(bytes_transferred == 1);
  assert(sb.size() == 1);
  assert(s.check_buffers(0, sb.data(), 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_exactly(1));
  assert(bytes_transferred == 1);
  assert(sb.size() == 1);
  assert(s.check_buffers(1234, sb.data(), 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_exactly(1));
  assert(bytes_transferred == 1);
  assert(sb.size() == 1);
  assert(s.check_buffers(0, sb.data(), 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_exactly(1));
  assert(bytes_transferred == 1);
  assert(sb.size() == 1);
  assert(s.check_buffers(1234, sb.data(), 1));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_exactly(10));
  assert(bytes_transferred == 10);
  assert(sb.size() == 10);
  assert(s.check_buffers(0, sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_exactly(10));
  assert(bytes_transferred == 10);
  assert(sb.size() == 10);
  assert(s.check_buffers(1234, sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_exactly(10));
  assert(bytes_transferred == 10);
  assert(sb.size() == 10);
  assert(s.check_buffers(0, sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_exactly(10));
  assert(bytes_transferred == 10);
  assert(sb.size() == 10);
  assert(s.check_buffers(1234, sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_exactly(10));
  assert(bytes_transferred == 10);
  assert(sb.size() == 10);
  assert(s.check_buffers(0, sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_exactly(10));
  assert(bytes_transferred == 10);
  assert(sb.size() == 10);
  assert(s.check_buffers(1234, sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_exactly(42));
  assert(bytes_transferred == 42);
  assert(sb.size() == 42);
  assert(s.check_buffers(0, sb.data(), 42));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_exactly(42));
  assert(bytes_transferred == 42);
  assert(sb.size() == 42);
  assert(s.check_buffers(1234, sb.data(), 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_exactly(42));
  assert(bytes_transferred == 42);
  assert(sb.size() == 42);
  assert(s.check_buffers(0, sb.data(), 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_exactly(42));
  assert(bytes_transferred == 42);
  assert(sb.size() == 42);
  assert(s.check_buffers(1234, sb.data(), 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_exactly(42));
  assert(bytes_transferred == 42);
  assert(sb.size() == 42);
  assert(s.check_buffers(0, sb.data(), 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_exactly(42));
  assert(bytes_transferred == 42);
  assert(sb.size() == 42);
  assert(s.check_buffers(1234, sb.data(), 42));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb, old_style_transfer_all);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    old_style_transfer_all);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb, old_style_transfer_all);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    old_style_transfer_all);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb, old_style_transfer_all);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    old_style_transfer_all);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb, short_transfer);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb, short_transfer);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 0, sb, short_transfer);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read_at(s, 1234, sb, short_transfer);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(1234, sb.data(), sizeof(read_data)));
}

// Test mutable buffers_1 read_at
void test_10() {
  boost::asio::io_service ios;
  stream_access_device s(ios);
  char read_buf[sizeof(read_data)];
  boost::asio::mutable_buffers_1 buffers
    = boost::asio::buffer(read_buf, sizeof(read_buf));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  boost::system::error_code ec;
  std::size_t bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_all(), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_data));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_all(), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_all(), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_data));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_all(), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_all(), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_all());
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(1), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(1), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(1), ec);
  assert(!ec);
  assert(bytes_transferred == 1);
  assert(s.check_buffers(0, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(1), ec);
  assert(!ec);
  assert(bytes_transferred == 1);
  assert(s.check_buffers(1234, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(1), ec);
  assert(!ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(1), ec);
  assert(!ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(50);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(1), ec);
  assert(!ec);
  assert(bytes_transferred == 50);
  assert(s.check_buffers(0, buffers, 50));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(50);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(1), ec);
  assert(!ec);
  assert(bytes_transferred == 50);
  assert(s.check_buffers(1234, buffers, 50));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(1), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(1), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(1), ec);
  assert(!ec);
  assert(bytes_transferred == 1);
  assert(s.check_buffers(0, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(1), ec);
  assert(!ec);
  assert(bytes_transferred == 1);
  assert(s.check_buffers(1234, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(1), ec);
  assert(!ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(1), ec);
  assert(!ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(1), ec);
  assert(!ec);
  assert(bytes_transferred = 1);
  assert(s.check_buffers(0, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(1), ec);
  assert(!ec);
  assert(bytes_transferred == 1);
  assert(s.check_buffers(1234, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(1), ec);
  assert(!ec);
  assert(bytes_transferred == 1);
  assert(s.check_buffers(0, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(1), ec);
  assert(!ec);
  assert(bytes_transferred == 1);
  assert(s.check_buffers(1234, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(1), ec);
  assert(!ec);
  assert(bytes_transferred == 1);
  assert(s.check_buffers(0, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(1), ec);
  assert(!ec);
  assert(bytes_transferred == 1);
  assert(s.check_buffers(1234, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(10), ec);
  assert(!ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(10), ec);
  assert(!ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(10), ec);
  assert(!ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(10), ec);
  assert(!ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(10), ec);
  assert(!ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(10), ec);
  assert(!ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(42), ec);
  assert(!ec);
  assert(bytes_transferred == 42);
  assert(s.check_buffers(0, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(42), ec);
  assert(!ec);
  assert(bytes_transferred == 42);
  assert(s.check_buffers(1234, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(42));
  assert(!ec);
  assert(bytes_transferred == 42);
  assert(s.check_buffers(0, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(42));
  assert(!ec);
  assert(bytes_transferred == 42);
  assert(s.check_buffers(1234, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(42));
  assert(!ec);
  assert(bytes_transferred == 42);
  assert(s.check_buffers(0, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(42));
  assert(!ec);
  assert(bytes_transferred == 42);
  assert(s.check_buffers(1234, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    old_style_transfer_all, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    old_style_transfer_all, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    old_style_transfer_all, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    old_style_transfer_all, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    old_style_transfer_all, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    old_style_transfer_all, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    short_transfer, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    short_transfer, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    short_transfer, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    short_transfer, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));
}

// Test vector buffers read_at
void test_11() {
  boost::asio::io_service ios;
  stream_access_device s(ios);
  char read_buf[sizeof(read_data)];
  std::vector<boost::asio::mutable_buffer> buffers;
  buffers.push_back(boost::asio::buffer(read_buf, 32));
  buffers.push_back(boost::asio::buffer(read_buf) + 32);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  boost::system::error_code ec;
  size_t bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_all(), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_all(), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_all(), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_all(), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_all(), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_all(), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(1), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(1), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(1), ec);
  assert(!ec);
  assert(bytes_transferred == 1);
  assert(s.check_buffers(0, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(1), ec);
  assert(!ec);
  assert(bytes_transferred == 1);
  assert(s.check_buffers(1234, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(1), ec);
  assert(!ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(1), ec);
  assert(!ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(10), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(10), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(10), ec);
  assert(!ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(10), ec);
  assert(!ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(10), ec);
  assert(!ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(10), ec);
  assert(!ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_at_least(42), ec);
  assert(!ec);
  assert(bytes_transferred == 50);
  assert(s.check_buffers(0, buffers, 50));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_at_least(42), ec);
  assert(!ec);
  assert(bytes_transferred == 50);
  assert(s.check_buffers(1234, buffers, 50));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(1), ec);
  assert(!ec);
  assert(bytes_transferred == 1);
  assert(s.check_buffers(0, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(1), ec);
  assert(!ec);
  assert(bytes_transferred == 1);
  assert(s.check_buffers(1234, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(1), ec);
  assert(!ec);
  assert(bytes_transferred == 1);
  assert(s.check_buffers(0, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(1), ec);
  assert(!ec);
  assert(bytes_transferred == 1);
  assert(s.check_buffers(1234, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(1), ec);
  assert(!ec);
  assert(bytes_transferred == 1);
  assert(s.check_buffers(0, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_data));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(1), ec);
  assert(!ec);
  assert(bytes_transferred == 1);
  assert(s.check_buffers(1234, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(10), ec);
  assert(!ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(10), ec);
  assert(!ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(10), ec);
  assert(!ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(10), ec);
  assert(!ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(10), ec);
  assert(!ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(10), ec);
  assert(!ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(50);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(10), ec);
  assert(!ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(50);
  memset(read_buf, 0, sizeof(read_data));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(10), ec);
  assert(!ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(50), ec);
  assert(!ec);
  assert(bytes_transferred == 50);
  assert(s.check_buffers(0, buffers, 50));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(50), ec);
  assert(!ec);
  assert(bytes_transferred == 50);
  assert(s.check_buffers(1234, buffers, 50));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(50), ec);
  assert(!ec);
  assert(bytes_transferred == 50);
  assert(s.check_buffers(0, buffers, 50));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(50), ec);
  assert(!ec);
  assert(bytes_transferred == 50);
  assert(s.check_buffers(1234, buffers, 50));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    boost::asio::transfer_exactly(50));
  assert(!ec);
  assert(bytes_transferred == 50);
  assert(s.check_buffers(0, buffers, 50));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    boost::asio::transfer_exactly(50));
  assert(!ec);
  assert(bytes_transferred == 50);
  assert(s.check_buffers(1234, buffers, 50));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    old_style_transfer_all, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    old_style_transfer_all, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    old_style_transfer_all, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    old_style_transfer_all, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    old_style_transfer_all, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    old_style_transfer_all, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    short_transfer, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    short_transfer, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    short_transfer, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    short_transfer, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, buffers,
    short_transfer, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, buffers,
    short_transfer, ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, buffers, sizeof(read_data)));
}

// Test streambuf read_at
void test_12() {
  boost::asio::io_service ios;
  stream_access_device s(ios);
  boost::asio::streambuf sb(sizeof(read_data));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  boost::system::error_code ec;
  std::size_t bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_all(), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_all(), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_all(), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_all(), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_all(), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_all(), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_at_least(1), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_at_least(1), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_at_least(1), ec);
  assert(!ec);
  assert(bytes_transferred == 1);
  assert(sb.size() == 1);
  assert(s.check_buffers(0, sb.data(), 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_at_least(1), ec);
  assert(!ec);
  assert(bytes_transferred == 1);
  assert(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_at_least(1), ec);
  assert(!ec);
  assert(bytes_transferred == 1);
  assert(s.check_buffers(0, sb.data(), 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_at_least(1), ec);
  assert(!ec);
  assert(bytes_transferred == 1);
  assert(s.check_buffers(1234, sb.data(), 1));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_at_least(10), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_at_least(10), ec);
  assert(!ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_at_least(10), ec);
  assert(!ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(0, sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_at_least(10));
  assert(!ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(1234, sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_at_least(10));
  assert(!ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(0, sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1324, sb,
    boost::asio::transfer_at_least(10));
  assert(!ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(1234, sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 0, sb,
    boost::asio::transfer_at_least(48));
  assert(!ec);
  assert(bytes_transferred == 50);
  assert(s.check_buffers(0, sb.data(), 50));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read_at(s, 1234, sb,
    boost::asio::transfer_at_least(48));
  assert(!ec);
  assert(bytes_transferred == 50);
  assert(s.check_buffers(1234, sb.data(), 50));
}

auto main() -> decltype(0) {
  test_1(); test_2(); test_3(); test_4(); test_5(); test_6(); test_7();
  test_8(); test_9(); test_10(); test_11();
  return 0;
}
