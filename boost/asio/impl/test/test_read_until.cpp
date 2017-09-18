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

  template <typename mutable_buffers, typename handler>
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

static const char read_data[] =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

void test_read_until() {
  boost::asio::io_service ios;
  stream s(ios);
  boost::asio::streambuf sb1;
  boost::asio::streambuf sb2(25);
  boost::system::error_code ec;

  {
    s.reset(read_data, sizeof(read_data));
    sb1.consume(sb1.size());
    std::size_t length = boost::asio::read_until(s, sb1, 'Z');
    assert(length == 26);

    s.reset(read_data, sizeof(read_data));
    s.next_read_length(1);
    sb1.consume(sb1.size());
    length = boost::asio::read_until(s, sb1, 'Z');
    assert(length == 26);

    s.reset(read_data, sizeof(read_data));
    s.next_read_length(10);
    sb1.consume(sb1.size());
    length = boost::asio::read_until(s, sb1, 'Z');

    s.reset(read_data, sizeof(read_data));
    sb1.consume(sb1.size());
    length = boost::asio::read_until(s, sb1, 'Z', ec);
    assert(!ec);
    assert(length == 26);

    s.reset(read_data, sizeof(read_data));
    s.next_read_length(1);
    sb1.consume(sb1.size());
    length = boost::asio::read_until(s, sb1, 'Z', ec);
    assert(!ec);
    assert(length == 26);

    s.reset(read_data, sizeof(read_data));
    s.next_read_length(10);
    sb1.consume(sb1.size());
    length = boost::asio::read_until(s, sb1, 'Z', ec);
    assert(!ec);
    assert(length == 26);

    s.reset(read_data, sizeof(read_data));
    sb1.consume(sb1.size());
    length = boost::asio::read_until(s, sb1, 'Z', ec);
    assert(ec != boost::asio::error::not_found);
    assert(length == 26);

    s.reset(read_data, sizeof(read_data));
    s.next_read_length(1);
    sb1.consume(sb1.size());
    length = boost::asio::read_until(s, sb1, 'Z', ec);
    assert(ec != boost::asio::error::not_found);
    assert(length == 26);

    s.reset(read_data, sizeof(read_data));
    s.next_read_length(10);
    sb1.consume(sb1.size());
    length = boost::asio::read_until(s, sb1, 'Z', ec);
    assert(ec != boost::asio::error::not_found);
    assert(length == 26);

    s.reset(read_data, sizeof(read_data));
    sb2.consume(sb2.size());
    length = boost::asio::read_until(s, sb2, 'Z', ec);
    assert(ec == boost::asio::error::not_found);
    assert(length == 0);

    s.reset(read_data, sizeof(read_data));
    s.next_read_length(1);
    sb2.consume(sb2.size());
    length = boost::asio::read_until(s, sb2, 'Z', ec);
    assert(ec == boost::asio::error::not_found);
    assert(length == 0);

    s.reset(read_data, sizeof(read_data));
    s.next_read_length(10);
    sb2.consume(sb2.size());
    length = boost::asio::read_until(s, sb2, 'Z', ec);
    assert(ec == boost::asio::error::not_found);
    assert(length == 0);

    s.reset(read_data, sizeof(read_data));
    sb2.consume(sb2.size());
    length = boost::asio::read_until(s, sb2, 'Y', ec);
    assert(ec != boost::asio::error::not_found);
    assert(length == 25);

    s.reset(read_data, sizeof(read_data));
    s.next_read_length(1);
    sb2.consume(sb2.size());
    length = boost::asio::read_until(s, sb2, 'Y', ec);
    assert(ec != boost::asio::error::not_found);
    assert(length == 25);

    s.reset(read_data, sizeof(read_data));
    s.next_read_length(10);
    sb2.consume(sb2.size());
    length = boost::asio::read_until(s, sb2, 'Y', ec);
    assert(ec != boost::asio::error::not_found);
    assert(length == 25);
  }
}

auto main() -> decltype(0) {
  test_read_until();
  return 0;
}
