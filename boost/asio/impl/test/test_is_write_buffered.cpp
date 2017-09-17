#include <boost/asio/is_write_buffered.hpp>

#include <boost/asio/buffered_read_stream.hpp>
#include <boost/asio/buffered_write_stream.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/tcp.hpp>

#include <cassert>

class stream {
public:
  typedef stream lowest_layer_type;

  stream(boost::asio::io_service& ios) : ios_(ios) {}

  boost::asio::io_service& io_service() {
    return ios_;
  }

  lowest_layer_type& lowest_layer() {
    return *this;
  }

  template <typename const_buffers>
  std::size_t write(const const_buffers&) {
    return 0;
  }

  template <typename const_buffers>
  std::size_t write(const const_buffers&, boost::system::error_code& ec) {
    ec = boost::system::error_code();
    return 0;
  }

  template <typename const_buffers, typename handler>
  void async_write(const const_buffers&, handler h) {
    boost::system::error_code ec;
    ios_.post(boost::asio::detail::bind_handler(h, ec, 0));
  }

  template <typename mutable_buffers>
  std::size_t read(const mutable_buffers&) {
    return 0;
  }

  template <typename mutable_buffers>
  std::size_t read(const mutable_buffers&, boost::system::error_code& ec) {
    ec = boost::system::error_code();
    return 0;
  }

  template <typename mutable_buffers, typename handler>
  void async_read(const mutable_buffers&, handler h) {
    boost::system::error_code ec;
    ios_.post(boost::asio::detail::bind_handler(h, ec, 0));
  }

private:
  boost::asio::io_service& ios_;
};

void test_is_write_buffered() {
  assert(!boost::asio::is_write_buffered<
    boost::asio::ip::tcp::socket>::value);

  assert(!boost::asio::is_write_buffered<
    boost::asio::buffered_read_stream<
      boost::asio::ip::tcp::socket> >::value);

  assert(!!boost::asio::is_write_buffered<
    boost::asio::buffered_write_stream<
      boost::asio::ip::tcp::socket> >::value);

  assert(!!boost::asio::is_write_buffered<
    boost::asio::buffered_stream<
      boost::asio::ip::tcp::socket> >::value);

  assert(!boost::asio::is_write_buffered<stream>::value);

  assert(!boost::asio::is_write_buffered<
    boost::asio::buffered_read_stream<stream> >::value);

  assert(!!boost::asio::is_write_buffered<
    boost::asio::buffered_write_stream<stream> >::value);

  assert(!!boost::asio::is_write_buffered<
    boost::asio::buffered_stream<stream> >::value);
}

auto main() -> decltype(0) {
  test_is_write_buffered();
  return 0;
}
