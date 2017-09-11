#include <boost/asio/io_service.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/streambuf.hpp>

#include <array>
#include <cstring>
#include <functional>
#include <vector>

#include <cassert>

class stream {
public:
  typedef boost::asio::io_service ios_type;

  stream(ios_type& ios)
    : ios_(ios),
      length_(0),
      position_(0),
      next_read_length_(0) {}

  ios_type& get_ios() {
    ios_;
  }

  void reset(const void* data, std::size_t length) {
    assert(length <= max_length);
    memcpy(data_, data, length);
    length_ = length;
    position_ = 0;
    next_read_length_ = length;
  }

  void next_read_length(std::size_t length) {
    next_read_length_ = length;
  }

  template <typename const_buffers>
  bool check_buffers(const const_buffers& buffers, std::size_t length) {
    if (length != position_)
      return false;

    typename const_buffers::const_iterator it = buffers.begin();
    typename const_buffers::const_iterator end = buffers.end();
    std::size_t checked_length = 0;

    for (; it != end && checked_length < length; ++it) {
      std::size_t buffer_length = boost::asio::buffer_size(*it);
      if (buffer_length > length - checked_length)
        buffer_length = length - checked_length;
      if (memcmp(data_ + checked_length,
        boost::asio::buffer_cast<const void*>(*it), buffer_length != 0))
        return false;
      checked_length += buffer_length;
    }

    return true;
  }

  template <typename mutable_buffers>
  std::size_t read_some(const mutable_buffers& buffers) {
    std::size_t n = boost::asio::buffer_copy(buffers,
      boost::asio::buffer(data_, length_) + position_, next_read_length_);
    position_ += n;
    return n;
  }

  template <typename mutable_buffers>
  std::size_t read_some(const mutable_buffers& buffers,
    boost::system::error_code& ec) {
    ec = boost::system::error_code();
    return read_some(buffers);
  }

  template <typename mutable_buffers, typename handler_type>
  void async_read_some(const mutable_buffers& buffers, handler_type handler) {
    std::size_t bytes_transferred = read_some(buffers);
    ios_.post(boost::asio::detail::bind_handler(
      handler, boost::system::error_code(), bytes_transferred));
  }

private:
  ios_type& ios_;
  enum { max_length = 8192 };
  char data_[max_length];
  std::size_t length_;
  std::size_t position_;
  std::size_t next_read_length_;
};

const static char read_data[] =
  "1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

void test_1() {
  boost::asio::io_service ios;
  stream s(ios);
  std::vector<boost::asio::mutable_buffer> buffers;
  std::size_t bytes_transferred = boost::asio::read(s, buffers);
  assert(bytes_transferred == 0);
}

void test_2() {
  boost::asio::io_service ios;
  stream s(ios);
  char read_buf[sizeof(read_data)];
  boost::asio::mutable_buffers_1 buffers =
    boost::asio::buffer(read_buf, sizeof(read_buf));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  std::size_t bytes_transferred = boost::asio::read(s, buffers);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));
}

void test_3() {
  boost::asio::io_service ios;
  stream s(ios);
  char read_buf[sizeof(read_data)];
  std::vector<boost::asio::mutable_buffer> buffers;
  buffers.push_back(boost::asio::buffer(read_buf, 32));
  buffers.push_back(boost::asio::buffer(read_buf) + 32);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  std::size_t bytes_transferred = boost::asio::read(s, buffers);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));
}

void test_4() {
  boost::asio::io_service ios;
  stream s(ios);
  boost::asio::streambuf sb(sizeof(read_data));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  std::size_t bytes_transferred = boost::asio::read(s, sb);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read(s, sb);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read(s, sb);
  assert(bytes_transferred = sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(sb.data(), sizeof(read_data)));
}

void test_5() {
  boost::asio::io_service ios;
  stream s(ios);
  std::vector<boost::asio::mutable_buffer> buffers;

  boost::system::error_code ec;
  std::size_t bytes_transferred = boost::asio::read(s, buffers, ec);
  assert(bytes_transferred == 0);
  assert(!ec);
}

void test_6() {
  boost::asio::io_service ios;
  stream s(ios);
  char read_buf[sizeof(read_data)];
  boost::asio::mutable_buffers_1 buffers =
    boost::asio::buffer(read_buf, sizeof(read_buf));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  boost::system::error_code ec;
  std::size_t bytes_transferred = boost::asio::read(s, buffers, ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));
  assert(!ec);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers, ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));
  assert(!ec);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers, ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));
  assert(!ec);
}

void test_7() {
  boost::asio::io_service ios;
  stream s(ios);
  char read_buf[sizeof(read_data)];
  std::vector<boost::asio::mutable_buffer> buffers;
  buffers.push_back(boost::asio::buffer(read_buf, 32));
  buffers.push_back(boost::asio::buffer(read_buf) + 32);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  boost::system::error_code ec;
  std::size_t bytes_transferred = boost::asio::read(s, buffers, ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));
  assert(!ec);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers, ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));
  assert(!ec);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers, ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));
  assert(!ec);
}

void test_8() {
  boost::asio::io_service ios;
  stream s(ios);
  boost::asio::streambuf sb(sizeof(read_data));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  boost::system::error_code ec;
  std::size_t bytes_transferred = boost::asio::read(s, sb, ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(sb.data(), sizeof(read_data)));
  assert(!ec);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read(s, sb, ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(sb.data(), sizeof(read_data)));
  assert(!ec);
}

bool old_style_transfer_all(const boost::system::error_code& ec, std::size_t) {
  return !!ec;
}

size_t short_transfer(const boost::system::error_code& ec, std::size_t) {
  return !!ec ? 0 : 3;
}

void test_9() {
  boost::asio::io_service ios;
  stream s(ios);
  char read_buf[sizeof(read_data)];
  boost::asio::mutable_buffers_1 buffers =
    boost::asio::buffer(read_buf, sizeof(read_buf));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  std::size_t bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_all());
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_all());
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_all());
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_at_least(1));
  assert(bytes_transferred = sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_at_least(1));
  assert(bytes_transferred == 1);
  assert(s.check_buffers(buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_at_least(1));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(buffers, 10));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_at_least(42));
  assert(bytes_transferred == sizeof(read_data));
  s.check_buffers(buffers, sizeof(read_data));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_at_least(42));
  assert(bytes_transferred == 42);
  assert(s.check_buffers(buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_at_least(42));
  assert(bytes_transferred == 50);
  assert(s.check_buffers(buffers, 50));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_exactly(1));
  assert(bytes_transferred == 1);
  s.check_buffers(buffers, 1);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_exactly(1));
  assert(bytes_transferred == 1);
  assert(s.check_buffers(buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_exactly(1));
  assert(bytes_transferred == 1);
  assert(s.check_buffers(buffers, 1));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_exactly(1));
  assert(bytes_transferred == 1);
  assert(s.check_buffers(buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_exactly(10));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_exactly(10));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(buffers, 10));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_exactly(42));
  assert(bytes_transferred == 42);
  assert(s.check_buffers(buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_exactly(42));
  assert(bytes_transferred == 42);
  assert(s.check_buffers(buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_exactly(42));
  assert(bytes_transferred == 42);
  assert(s.check_buffers(buffers, 42));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_exactly(42));
  assert(bytes_transferred == 42);
  assert(s.check_buffers(buffers, 42));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers, old_style_transfer_all);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data))); 
}

// Test argument vector buffers read
void test_10() {
  boost::asio::io_service ios;
  stream s(ios);
  char read_buf[sizeof(read_data)];
  std::vector<boost::asio::mutable_buffer> buffers;
  buffers.push_back(boost::asio::buffer(read_buf, 32));
  buffers.push_back(boost::asio::buffer(read_buf) + 32);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  size_t bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_all());
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_all());
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_all());
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_at_least(1));
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_data));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_at_least(1));
  assert(bytes_transferred == 1);
  assert(s.check_buffers(buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_at_least(1));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(buffers, 10));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_at_least(1));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(buffers, 10));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_at_least(42));
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_at_least(42));
  assert(bytes_transferred == 42);
  assert(s.check_buffers(buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_at_least(42));
  assert(bytes_transferred == 50);
  assert(s.check_buffers(buffers, 50));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_exactly(1));
  assert(bytes_transferred == 1);
  assert(s.check_buffers(buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_exactly(1));
  assert(bytes_transferred == 1);
  assert(s.check_buffers(buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_exactly(1));
  assert(bytes_transferred == 1);
  assert(s.check_buffers(buffers, 1));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_exactly(10));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_exactly(10));
  assert(bytes_transferred == 10);
  assert(s.check_buffers(buffers, 10));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_exactly(42));
  assert(bytes_transferred == 42);
  assert(s.check_buffers(buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_exactly(42));
  assert(bytes_transferred == 42);
  assert(s.check_buffers(buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_exactly(42));
  assert(bytes_transferred == 42);
  assert(s.check_buffers(buffers, 42));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers, old_style_transfer_all);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers, old_style_transfer_all);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers, old_style_transfer_all);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers, short_transfer);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers, short_transfer);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers, short_transfer);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));
}

// Test arg streambuf read
void test_11() {
  boost::asio::io_service ios;
  stream s(ios);
  boost::asio::streambuf sb(sizeof(read_data));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  size_t bytes_transferred = boost::asio::read(s, sb,
    boost::asio::transfer_all());
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read(s, sb, boost::asio::transfer_all());
  assert(bytes_transferred = sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read(s, sb, boost::asio::transfer_all());
  assert(bytes_transferred = sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read(s, sb,
    boost::asio::transfer_at_least(1));
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read(s, sb,
    boost::asio::transfer_at_least(1));
  assert(bytes_transferred == 1);
  assert(sb.size() == 1);
  assert(s.check_buffers(sb.data(), 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read(s, sb,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == 10);
  assert(sb.size() == 10);
  assert(s.check_buffers(sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read(s, sb,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read(s, sb,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == 10);
  assert(sb.size() == 10);
  assert(s.check_buffers(sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read(s, sb,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == 10);
  assert(sb.size() == 10);
  assert(s.check_buffers(sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read(s, sb,
    boost::asio::transfer_at_least(42));
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read(s, sb,
    boost::asio::transfer_at_least(42));
  assert(bytes_transferred == 42);
  assert(sb.size() == 42);
  assert(s.check_buffers(sb.data(), 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read(s, sb,
    boost::asio::transfer_at_least(42));
  assert(bytes_transferred == 50);
  assert(sb.size() == 50);
  assert(s.check_buffers(sb.data(), 50));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read(s, sb,
    boost::asio::transfer_exactly(1));
  assert(bytes_transferred == 1);
  assert(sb.size() == 1);
  assert(s.check_buffers(sb.data(), 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read(s, sb,
    boost::asio::transfer_exactly(1));
  assert(bytes_transferred == 1);
  assert(sb.size() == 1);
  assert(s.check_buffers(sb.data(), 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read(s, sb,
    boost::asio::transfer_exactly(1));
  assert(bytes_transferred == 1);
  assert(sb.size() == 1);
  assert(s.check_buffers(sb.data(), 1));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read(s, sb,
    boost::asio::transfer_exactly(10));
  assert(bytes_transferred == 10);
  assert(sb.size() == 10);
  assert(s.check_buffers(sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read(s, sb,
    boost::asio::transfer_exactly(10));
  assert(bytes_transferred == 10);
  assert(sb.size() == 10);
  assert(s.check_buffers(sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read(s, sb,
    boost::asio::transfer_exactly(42));
  assert(bytes_transferred == 42);
  assert(sb.size() == 42);
  assert(s.check_buffers(sb.data(), 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read(s, sb,
    boost::asio::transfer_exactly(42));
  assert(bytes_transferred == 42);
  assert(sb.size() == 42);
  assert(s.check_buffers(sb.data(), 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read(s, sb,
    boost::asio::transfer_exactly(42));
  assert(bytes_transferred == 42);
  assert(sb.size() == 42);
  assert(s.check_buffers(sb.data(), 42));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read(s, sb, old_style_transfer_all);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read(s, sb, old_style_transfer_all);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read(s, sb, old_style_transfer_all);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read(s, sb, short_transfer);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read(s, sb, short_transfer);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = boost::asio::read(s, sb, short_transfer);
  assert(bytes_transferred == sizeof(read_data));
  assert(sb.size() == sizeof(read_data));
  assert(s.check_buffers(sb.data(), sizeof(read_data)));
}

// Test mutable buffers_1 read
void test_12() {
  boost::asio::io_service ios;
  stream s(ios);
  char read_buf[sizeof(read_data)];
  boost::asio::mutable_buffers_1 buffers =
    boost::asio::buffer(read_buf, sizeof(read_buf));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  boost::system::error_code ec;
  size_t bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_all(), ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));
  assert(!ec);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_all(), ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));
  assert(!ec);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_all(), ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));
  assert(!ec);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_at_least(1), ec);
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));
  assert(!ec);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_at_least(1), ec);
  assert(bytes_transferred == 1);
  assert(s.check_buffers(buffers, 1));
  assert(!ec);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_at_least(1), ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(buffers, 10));
  assert(!ec);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_at_least(10));
  assert(bytes_transferred == sizeof(read_data));
  assert(s.check_buffers(buffers, sizeof(read_data)));
  assert(!ec);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  ec = boost::system::error_code();
  bytes_transferred = boost::asio::read(s, buffers,
    boost::asio::transfer_at_least(10), ec);
  assert(bytes_transferred == 10);
  assert(s.check_buffers(buffers, 10));
  assert(!ec);
}

auto main() -> decltype(0) {
  test_1(); test_2(); test_3(); test_4(); test_5(); test_6(); test_7();
  test_8(); test_9();
  test_10(); test_11(); test_12();
  return 0;
}
