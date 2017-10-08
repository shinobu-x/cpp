#include <boost/asio/serial_port.hpp>

#include <boost/asio/io_service.hpp>

#include "async_result.hpp"

namespace compile {

void write_some_handler(const boost::system::error_code&, std::size_t) {}

void read_some_handler(const boost::system::error_code&, std::size_t) {}

void test_1() {
  try {
    boost::asio::io_service ios;
    char mutable_char_buffer[128] = "";
    const char const_char_buffer[128] = "";
    boost::asio::serial_port::baud_rate serial_port_option;
    boost::system::error_code ec;

    boost::asio::serial_port port1(ios);
    boost::asio::serial_port port2(ios, "null");
    boost::asio::serial_port::native_handle_type native_port1 =
      port1.native_handle();
    boost::asio::serial_port port3(ios, native_port1);

    boost::asio::serial_port port4(std::move(port1));
    port1 = boost::asio::serial_port(ios);
    port1 = std::move(port2);

    boost::asio::io_service& ios_ref = port1.get_io_service();
    (void)ios_ref;

    boost::asio::serial_port::lowest_layer_type& lowest_layer =
      port1.lowest_layer();
    (void)lowest_layer;

    const boost::asio::serial_port& port5 = port1;
    const boost::asio::serial_port& lowest_layer2 = port5.lowest_layer();
    (void)lowest_layer2;

    port1.open("null");
    port1.open("null", ec);

    boost::asio::serial_port::native_handle_type native_port2 =
      port1.native_handle();
    (void)native_port2;

    bool is_open = port1.is_open();
    (void)is_open;

    port1.close();
    port1.close(ec);

    boost::asio::serial_port::native_type native_port3 = port1.native();
    (void)native_port3;

    port1.cancel();
    port1.cancel(ec);

    port1.set_option(serial_port_option);
    port1.set_option(serial_port_option, ec);

    port1.get_option(serial_port_option);
    port1.get_option(serial_port_option, ec);

    port1.send_break();
    port1.send_break(ec);

    port1.write_some(boost::asio::buffer(mutable_char_buffer));
    port1.write_some(boost::asio::buffer(mutable_char_buffer), ec);
    port1.write_some(boost::asio::buffer(const_char_buffer));
    port1.write_some(boost::asio::buffer(const_char_buffer), ec);

    port1.async_write_some(
      boost::asio::buffer(mutable_char_buffer), &write_some_handler);
    port1.async_write_some(
      boost::asio::buffer(const_char_buffer), &write_some_handler);
    int l1 = port1.async_write_some(
      boost::asio::buffer(mutable_char_buffer), lazy_handler());
    (void)l1;
    int l2 = port1.async_write_some(
      boost::asio::buffer(const_char_buffer), lazy_handler());
    (void)l2;

    port1.read_some(boost::asio::buffer(mutable_char_buffer));
    port1.read_some(boost::asio::buffer(mutable_char_buffer), ec);

    port1.async_read_some(
      boost::asio::buffer(mutable_char_buffer), lazy_handler());
    int l3 = port1.async_read_some(
      boost::asio::buffer(mutable_char_buffer), lazy_handler());
    (void)l3;
  } catch (...) {}
}
} // compile

auto main() -> decltype(0) {
  compile::test_1();
  return 0;
}
