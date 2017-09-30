// -lcrypto -lssl

#include <boost/asio/ssl/stream.hpp>
#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>

#include <cassert>

#include "async_result.hpp"

namespace {

bool verify_callback(bool, boost::asio::ssl::verify_context&) {
  return false;
}

void handshake_handler(const boost::system::error_code&) {}

void buffered_handshake_handler(
  const boost::system::error_code&, std::size_t) {}

void shutdown_handler(const boost::system::error_code&) {}

void write_some_handler(const boost::system::error_code&, std::size_t) {}

void read_some_handler(const boost::system::error_code&, std::size_t) {}

void test_1() {
  try {
    boost::asio::io_service ios;
    char mutable_char_buffer[128] = "";
    const char const_char_buffer[128] = "";
    boost::asio::ssl::context context(ios, boost::asio::ssl::context::sslv23);
    lazy_handler lazy;
    boost::system::error_code ec;

    boost::asio::ssl::stream<boost::asio::ip::tcp::socket> stream1(
      ios, context);
    boost::asio::ip::tcp::socket socket1(ios, boost::asio::ip::tcp::v4());
    boost::asio::ssl::stream<boost::asio::ip::tcp::socket&> stream2(
      socket1, context);

    boost::asio::io_service& ios_ref = stream1.get_io_service();
    (void)ios_ref;

    SSL* ssl1 = stream1.native_handle();
    (void)ssl1;

    SSL* ssl2 = stream1.impl()->ssl;
    (void)ssl2;

    boost::asio::ssl::stream<boost::asio::ip::tcp::socket>::lowest_layer_type&
      lowest_layer = stream1.lowest_layer();
    (void)lowest_layer;

    const boost::asio::ssl::stream<boost::asio::ip::tcp::socket>& stream3 =
      stream1;
    const boost::asio::ssl::stream<
      boost::asio::ip::tcp::socket>::lowest_layer_type& lowest_layer2 =
      stream3.lowest_layer();
    (void)lowest_layer2;

    stream1.set_verify_mode(boost::asio::ssl::verify_none);
    stream1.set_verify_mode(boost::asio::ssl::verify_none, ec);
    stream1.set_verify_depth(1);
    stream1.set_verify_depth(1, ec);
    stream1.set_verify_callback(verify_callback);
    stream1.set_verify_callback(verify_callback, ec);

    stream1.handshake(boost::asio::ssl::stream_base::client);
    stream1.handshake(boost::asio::ssl::stream_base::client, ec);
    stream1.handshake(boost::asio::ssl::stream_base::server);
    stream1.handshake(boost::asio::ssl::stream_base::client, ec);

    stream1.handshake(boost::asio::ssl::stream_base::client,
      boost::asio::buffer(mutable_char_buffer));
    stream1.handshake(boost::asio::ssl::stream_base::client,
      boost::asio::buffer(mutable_char_buffer), ec);
    stream1.handshake(boost::asio::ssl::stream_base::client,
      boost::asio::buffer(const_char_buffer));
    stream1.handshake(boost::asio::ssl::stream_base::client,
      boost::asio::buffer(const_char_buffer), ec);

    stream1.handshake(boost::asio::ssl::stream_base::server,
      boost::asio::buffer(mutable_char_buffer));
    stream1.handshake(boost::asio::ssl::stream_base::server,
      boost::asio::buffer(mutable_char_buffer), ec);
    stream1.handshake(boost::asio::ssl::stream_base::server,
      boost::asio::buffer(const_char_buffer));
    stream1.handshake(boost::asio::ssl::stream_base::server,
      boost::asio::buffer(const_char_buffer), ec);

    stream1.async_handshake(
      boost::asio::ssl::stream_base::client, handshake_handler);
    stream1.async_handshake(
      boost::asio::ssl::stream_base::client,
      boost::asio::buffer(mutable_char_buffer), buffered_handshake_handler);
    stream1.async_handshake(
      boost::asio::ssl::stream_base::client,
      boost::asio::buffer(const_char_buffer), buffered_handshake_handler);
    stream1.async_handshake(
      boost::asio::ssl::stream_base::server, handshake_handler);
    stream1.async_handshake(
      boost::asio::ssl::stream_base::server,
      boost::asio::buffer(mutable_char_buffer), buffered_handshake_handler);
    stream1.async_handshake(
      boost::asio::ssl::stream_base::server,
      boost::asio::buffer(const_char_buffer), buffered_handshake_handler);
    int l1 = stream1.async_handshake(
      boost::asio::ssl::stream_base::client, lazy);
    (void)l1;
    int l2 = stream1.async_handshake(
      boost::asio::ssl::stream_base::client,
      boost::asio::buffer(mutable_char_buffer), lazy);
    (void)l2;
    int l3 = stream1.async_handshake(
      boost::asio::ssl::stream_base::client,
      boost::asio::buffer(const_char_buffer), lazy);
    (void)l3;
    int l4 = stream1.async_handshake(
      boost::asio::ssl::stream_base::server, lazy);
    (void)l4;
    int l5 = stream1.async_handshake(
      boost::asio::ssl::stream_base::server,
      boost::asio::buffer(mutable_char_buffer), lazy);
    (void)l5;
    int l6 = stream1.async_handshake(
      boost::asio::ssl::stream_base::server,
      boost::asio::buffer(const_char_buffer), lazy);
    (void)l6;

    stream1.shutdown();
    stream1.shutdown(ec);

    stream1.async_shutdown(shutdown_handler);
    int l7 = stream1.async_shutdown(lazy);
    (void)l7;

    stream1.write_some(boost::asio::buffer(mutable_char_buffer));
    stream1.write_some(boost::asio::buffer(mutable_char_buffer), ec);
    stream1.write_some(boost::asio::buffer(const_char_buffer));
    stream1.write_some(boost::asio::buffer(const_char_buffer), ec);

    stream1.async_write_some(
      boost::asio::buffer(mutable_char_buffer), write_some_handler);
    stream1.async_write_some(
      boost::asio::buffer(const_char_buffer), write_some_handler);
    int l8 = stream1.async_write_some(
      boost::asio::buffer(mutable_char_buffer), lazy);
    (void)l8;
    int l9 = stream1.async_write_some(
      boost::asio::buffer(const_char_buffer), lazy);

    stream1.read_some(boost::asio::buffer(mutable_char_buffer));
    stream1.read_some(boost::asio::buffer(mutable_char_buffer), ec);

    stream1.async_read_some(
      boost::asio::buffer(mutable_char_buffer), read_some_handler);
    int l10 = stream1.async_read_some(
      boost::asio::buffer(mutable_char_buffer), lazy);
    (void)l10;
  } catch (...) {}

}
} // namespace

auto main() -> decltype(0) {
  test_1();
  return 0;
} 
