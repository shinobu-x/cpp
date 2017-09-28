#include <boost/asio/posix/stream_descriptor.hpp>

#include <boost/asio/io_service.hpp>

#include "async_result.hpp"

namespace {

void write_some_handler(const boost::system::error_code&, std::size_t) {}

void read_some_handler(const boost::system::error_code&, std::size_t) {}

void test_1() {

  try {
    boost::asio::io_service ios;
    char mutable_char_buffer[128] = "";
    const char const_char_buffer[128] = "";
    boost::asio::posix::descriptor_base::bytes_readable io_control_command;
    lazy_handler lazy;
    boost::system::error_code ec;

    boost::asio::posix::stream_descriptor descriptor1(ios);
    int native_descriptor1 = -1;
    boost::asio::posix::stream_descriptor descriptor2(ios, native_descriptor1);

    boost::asio::posix::stream_descriptor descriptor3(
      std::move(descriptor3));

    descriptor1 = boost::asio::posix::stream_descriptor(ios);
    descriptor1 = std::move(descriptor2);

    boost::asio::io_service& ios_ref = descriptor1.get_io_service();
    (void)ios_ref;

    boost::asio::posix::stream_descriptor::lowest_layer_type& lowest_layer1 =
      descriptor1.lowest_layer();
    (void)lowest_layer1;

    const boost::asio::posix::stream_descriptor& descriptor4 = descriptor1;
    const boost::asio::posix::stream_descriptor::lowest_layer_type&
      lowest_layer2 = descriptor4.lowest_layer();
    (void)lowest_layer2;

    int native_descriptor2 = -1;
    descriptor1.assign(native_descriptor2);

    bool is_open = descriptor1.is_open();
    (void)is_open;

    descriptor1.close();
    descriptor1.close(ec);

    boost::asio::posix::stream_descriptor::native_type native_descriptor3 =
      descriptor1.native();
    (void)native_descriptor3;

    boost::asio::posix::stream_descriptor::native_handle_type
      native_descriptor4 = descriptor1.native_handle();
    (void)native_descriptor4;

    boost::asio::posix::stream_descriptor::native_handle_type
      native_descriptor5 = descriptor1.release();
    (void)native_descriptor5;

    descriptor1.cancel();
    descriptor1.cancel(ec);

    descriptor1.io_control(io_control_command);
    descriptor1.io_control(io_control_command, ec);

    bool non_blocking1 = descriptor1.non_blocking();
    (void)non_blocking1;
    descriptor1.non_blocking(true);
    descriptor1.non_blocking(false, ec);

    bool non_blocking2 = descriptor1.native_non_blocking();
    (void)non_blocking2;
    descriptor1.native_non_blocking(true);
    descriptor1.native_non_blocking(false, ec);

    descriptor1.write_some(boost::asio::buffer(mutable_char_buffer));
    descriptor1.write_some(boost::asio::buffer(const_char_buffer));
    descriptor1.write_some(boost::asio::null_buffers());
    descriptor1.write_some(boost::asio::buffer(mutable_char_buffer), ec);
    descriptor1.write_some(boost::asio::buffer(const_char_buffer), ec);
    descriptor1.write_some(boost::asio::null_buffers(), ec);

    descriptor1.async_write_some(
      boost::asio::buffer(mutable_char_buffer), write_some_handler);
    descriptor1.async_write_some(
      boost::asio::buffer(const_char_buffer), write_some_handler);
    descriptor1.async_write_some(
      boost::asio::null_buffers(), write_some_handler);

    int l1 = descriptor1.async_write_some(
      boost::asio::buffer(mutable_char_buffer), lazy);
    (void)l1;
    int l2 = descriptor1.async_write_some(
      boost::asio::buffer(const_char_buffer), lazy);
    (void)l2;
    int l3 = descriptor1.async_write_some(
      boost::asio::null_buffers(), lazy);
    (void)l3;

    descriptor1.read_some(boost::asio::buffer(mutable_char_buffer));
    descriptor1.read_some(boost::asio::null_buffers());
    descriptor1.read_some(boost::asio::buffer(mutable_char_buffer), ec);
    descriptor1.read_some(boost::asio::null_buffers(), ec);

    descriptor1.async_read_some(
      boost::asio::buffer(mutable_char_buffer), read_some_handler);
    descriptor1.async_read_some(
      boost::asio::null_buffers(), read_some_handler);

    int l4 = descriptor1.async_read_some(
      boost::asio::buffer(mutable_char_buffer), lazy);
    (void)l4;
    int l5 = descriptor1.async_read_some(
      boost::asio::null_buffers(), lazy);
    (void)l5;

    {
      boost::asio::posix::stream_descriptor descriptor5(std::move(descriptor1));
      descriptor5.write_some(boost::asio::buffer(mutable_char_buffer));
      descriptor5.write_some(boost::asio::buffer(const_char_buffer));
      descriptor5.write_some(boost::asio::null_buffers());
      descriptor5.write_some(boost::asio::buffer(mutable_char_buffer), ec);
      descriptor5.write_some(boost::asio::buffer(const_char_buffer), ec);
      descriptor5.write_some(boost::asio::null_buffers(), ec);
      descriptor5.async_write_some(
        boost::asio::buffer(mutable_char_buffer), write_some_handler);
      descriptor5.async_write_some(
        boost::asio::buffer(const_char_buffer), write_some_handler);
      descriptor5.async_write_some(
        boost::asio::null_buffers(), write_some_handler);
      int l6 = descriptor5.async_write_some(
        boost::asio::buffer(mutable_char_buffer), lazy);
      (void)l6;
      int l7 = descriptor5.async_write_some(
        boost::asio::buffer(const_char_buffer), lazy);
      (void)l7;
      int l8 = descriptor5.async_write_some(
        boost::asio::null_buffers(), lazy);
      (void)l8;
      descriptor5.read_some(boost::asio::buffer(mutable_char_buffer));
      descriptor5.read_some(boost::asio::null_buffers());
      descriptor5.read_some(boost::asio::buffer(mutable_char_buffer), ec);
      descriptor5.read_some(boost::asio::null_buffers(), ec);
      descriptor5.async_read_some(
        boost::asio::buffer(mutable_char_buffer), read_some_handler);
      descriptor5.async_read_some(
        boost::asio::null_buffers(), read_some_handler);
      int l9 = descriptor5.async_read_some(
        boost::asio::buffer(mutable_char_buffer), lazy);
      (void)l9;
      int l10 = descriptor5.async_read_some(
        boost::asio::null_buffers(), lazy);
      (void)l10;
    }

  } catch (...) {}
}

} // namespace

auto main() -> decltype(0) {
  test_1();
  return 0;
}
