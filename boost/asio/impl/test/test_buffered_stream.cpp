#include <boost/asio/buffered_stream.hpp>

#include <boost/asio/buffer.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/tcp.hpp>

#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/system/system_error.hpp>

#include <cassert>
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

namespace runtime {
// Test sync operation
void test_1() {
  boost::asio::io_service ios;
  boost::asio::ip::tcp::acceptor acceptor(ios,
    boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), 0));
  boost::asio::ip::tcp::endpoint server_endpoint = acceptor.local_endpoint();
  server_endpoint.address(boost::asio::ip::address_v4::loopback());

  boost::asio::buffered_stream<boost::asio::ip::tcp::socket> client_socket(ios);
  client_socket.lowest_layer().connect(server_endpoint);

  boost::asio::buffered_stream<boost::asio::ip::tcp::socket> server_socket(ios);
  acceptor.accept(server_socket.lowest_layer());

  const char write_data[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  const boost::asio::const_buffer write_buf = boost::asio::buffer(write_data);

  std::size_t bytes_written = 0;

  while (bytes_written < sizeof(write_data)) {
    bytes_written += server_socket.write_some(
      boost::asio::buffer(write_buf + bytes_written));
    server_socket.flush();
  }

  char read_data[sizeof(write_data)];
  const boost::asio::mutable_buffer read_buf = boost::asio::buffer(read_data);

  std::size_t bytes_read = 0;

  while (bytes_read < sizeof(read_data)) {
    bytes_read += client_socket.read_some(
      boost::asio::buffer(read_buf + bytes_read));
  }

  assert(bytes_written == sizeof(write_data));
  assert(bytes_read == sizeof(read_data));
  assert(memcmp(write_data, read_data, sizeof(write_data)) == 0);

  server_socket.close();
  boost::system::error_code ec;
  bytes_read = client_socket.read_some(boost::asio::buffer(read_buf), ec);

  assert(bytes_read == 0);
  assert(ec == boost::asio::error::eof);

  client_socket.close(ec);
}

void handle_accept(const boost::system::error_code& ec) {
  assert(!ec);
}

void handle_write(const boost::system::error_code& ec,
  std::size_t bytes_transferred, std::size_t* total_bytes_written) {
  assert(!ec);

  if (ec)
    throw boost::system::system_error(ec);

  *total_bytes_written += bytes_transferred;
}

void handle_flush(const boost::system::error_code& ec) {
  assert(!ec);
}

void handle_read(const boost::system::error_code& ec,
  std::size_t bytes_transferred, std::size_t* total_bytes_read) {
  assert(!ec);

  if (ec)
    throw boost::system::system_error(ec);

  *total_bytes_read += bytes_transferred;
}

void handle_read_eof(const boost::system::error_code& ec,
  std::size_t bytes_transferred) {
  assert(ec == boost::asio::error::eof);
  assert(bytes_transferred == 0);
}

// Test async operation
void test_2() {
  boost::asio::io_service ios;
  boost::asio::ip::tcp::acceptor acceptor(ios,
    boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), 0));
  boost::asio::ip::tcp::endpoint server_endpoint = acceptor.local_endpoint();
  server_endpoint.address(boost::asio::ip::address_v4::loopback());

  boost::asio::buffered_stream<boost::asio::ip::tcp::socket> client_socket(ios);
  client_socket.lowest_layer().connect(server_endpoint);

  boost::asio::buffered_stream<boost::asio::ip::tcp::socket> server_socket(ios);
  acceptor.async_accept(server_socket.lowest_layer(), &handle_accept);
  ios.run();
  ios.reset();

  const char write_data[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  const boost::asio::const_buffer write_buf = boost::asio::buffer(write_data);

  std::size_t bytes_written = 0;

  while (bytes_written < sizeof(write_data)) {
    client_socket.async_write_some(
      boost::asio::buffer(write_buf + bytes_written),
      boost::bind(handle_write, _1, _2, &bytes_written));
    ios.run();
    ios.reset();
    client_socket.async_flush(boost::bind(handle_flush, _1));
    ios.run();
    ios.reset();
  }

  char read_data[sizeof(write_data)];
  const boost::asio::mutable_buffer read_buf = boost::asio::buffer(read_data);

  std::size_t bytes_read = 0;

  while (bytes_read < sizeof(read_data)) {
    server_socket.async_read_some(
      boost::asio::buffer(read_buf + bytes_read),
      boost::bind(handle_read, _1, _2, &bytes_read));
  ios.run();
  ios.reset();
  }

  assert(bytes_written == sizeof(write_data));
  assert(bytes_read == sizeof(read_data));
  assert(memcmp(write_data, read_data, sizeof(write_data)) == 0);

  bytes_written = 0;

  while (bytes_written < sizeof(write_data)) {
    server_socket.async_write_some(
      boost::asio::buffer(write_buf + bytes_written),
      boost::bind(handle_write, _1, _2, &bytes_written));
    ios.run();
    ios.reset();
    server_socket.async_flush(boost::bind(handle_flush, _1));
    ios.run();
    ios.reset();
  }

  bytes_read = 0;

  while (bytes_read < sizeof(read_data)) {
    client_socket.async_read_some(
      boost::asio::buffer(read_buf + bytes_read),
      boost::bind(handle_read, _1, _2, &bytes_read));
    ios.run();
    ios.reset();
  }

  assert(bytes_written == sizeof(write_data));
  assert(bytes_read == sizeof(read_data));
  assert(memcmp(write_data, read_data, sizeof(write_data)) == 0);

  server_socket.close();
  client_socket.async_read_some(boost::asio::buffer(read_buf), handle_read_eof);
}
} // namespace

auto main() -> decltype(0) {
  compile::test_1();
  runtime::test_1();
  runtime::test_2();
  return 0;
} 
