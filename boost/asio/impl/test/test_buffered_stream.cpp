#include <boost/asio/buffered_stream.hpp>

#include <boost/asio/buffer.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/tcp.hpp>

#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/system/system_error.hpp>

#include <cstring>

#include "async_result.hpp"

void write_some_handler(const boost::system::error_code&, std::size_t) {}

void flush_handler(const boost::system::error_code&, std::size_t) {}

void fill_handler(const boost::system::error_code&, std::size_t) {}

void read_some_handler(const boost::system::error_code&, std::size_t) {}

namespace compile {

void test_1() {
  try {
    boost::asio::io_service ios;
    char mutable_char_buffer[128] = "";
    const char const_char_buffer[128] = "";
    boost::array<boost::asio::mutable_buffer, 2> mutable_buffers = {{
      boost::asio::buffer(mutable_char_buffer, 10),
      boost::asio::buffer(mutable_char_buffer + 10, 10)
    }};
    boost::array<boost::asio::const_buffer, 2> const_buffers = {{
      boost::asio::buffer(const_char_buffer, 10),
      boost::asio::buffer(const_char_buffer + 10, 10)
    }};
    lazy_handler lazy;
    boost::system::error_code ec;

    boost::asio::buffered_stream<
      boost::asio::ip::tcp::socket> stream1(ios);
    boost::asio::buffered_stream<
      boost::asio::ip::tcp::socket> stream2(ios, 1024, 1024);

    boost::asio::io_service& ios_ref = stream1.get_io_service();
    (void)ios_ref;

    boost::asio::buffered_stream<
      boost::asio::ip::tcp::socket>::lowest_layer_type& lowest_layer =
      stream1.lowest_layer();
    (void)lowest_layer;

    stream1.write_some(boost::asio::buffer(mutable_char_buffer));
    stream1.write_some(boost::asio::buffer(mutable_char_buffer), ec);
    stream1.write_some(mutable_buffers);
    stream1.write_some(mutable_buffers, ec);
    stream1.write_some(boost::asio::buffer(const_char_buffer));
    stream1.write_some(boost::asio::buffer(const_char_buffer), ec);
    stream1.write_some(const_buffers);
    stream1.write_some(const_buffers, ec);
    stream1.write_some(boost::asio::null_buffers());
    stream1.write_some(boost::asio::null_buffers(), ec);

    stream1.async_write_some(
      boost::asio::buffer(mutable_char_buffer), &write_some_handler);
    stream1.async_write_some(mutable_buffers, &write_some_handler);
    stream1.async_write_some(
      boost::asio::buffer(const_char_buffer), &write_some_handler);
    stream1.async_write_some(const_buffers, &write_some_handler);
    stream1.async_write_some(
      boost::asio::null_buffers(), &write_some_handler);
    int l1 = stream1.async_write_some(
      boost::asio::buffer(mutable_char_buffer), lazy);
    (void)l1;
    int l2 = stream1.async_write_some(mutable_buffers, lazy);
    (void)l2;
    int l3 = stream1.async_write_some(
      boost::asio::buffer(const_char_buffer), lazy);
    (void)l3;
    int l4 = stream1.async_write_some(const_buffers, lazy);
    (void)l4;
    int l5 = stream1.async_write_some(boost::asio::null_buffers(), lazy);
    (void)l5;

    stream1.flush();
    stream1.flush(ec);

    stream1.async_flush(&flush_handler);
    int l6 = stream1.async_flush(lazy);
    (void)l6;

    stream1.fill();
    stream1.fill(ec);

    stream1.async_fill(&fill_handler);
    int l7 = stream1.async_fill(lazy);
    (void)l7;

    stream1.read_some(boost::asio::buffer(mutable_char_buffer));
    stream1.read_some(boost::asio::buffer(mutable_char_buffer), ec);
    stream1.read_some(mutable_buffers);
    stream1.read_some(mutable_buffers, ec);
    stream1.read_some(boost::asio::null_buffers());
    stream1.read_some(boost::asio::null_buffers(), ec);

    stream1.async_read_some(
      boost::asio::buffer(mutable_char_buffer), &read_some_handler);
    stream1.async_read_some(mutable_buffers, &read_some_handler);
    stream1.async_read_some(boost::asio::null_buffers(), &read_some_handler);

    int l8 = stream1.async_read_some(
      boost::asio::buffer(mutable_char_buffer), lazy);
    (void)l8;
    int l9 = stream1.async_read_some(mutable_buffers, lazy);
    (void)l9;
    int l10 = stream1.async_read_some(boost::asio::null_buffers(), lazy);
    (void)l10;
  } catch (...) {}
}
} // namespace

namespae runtime {
void test_1() {
  boost::asio::io_service ios;
  boost::asio::ip::tcp::acceptor acceptor(ios,
    boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), 0));
  boost::asio::ip::tcp::endpoint server_endpoint = acceptor.local_endpoint();
  server_endpoint.address(boost::asio::ip::address_v4::loopback());

  boost::asio::buffered_stream<boost::asio::ip::tcp::socket> client_socket(ios);
  client_socket.lowest_layer().connect(server_endpoint);
  acceptor.accept(server_socket.lowest_layer());

  const char write_data[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
}
} // namespace
auto main() -> decltype(0) {
  compile::test_1();
  return 0;
} 
