#include <boost/asio/serial_port_base.hpp>

#include <boost/asio/io_service.hpp>
#include <boost/asio/serial_port.hpp>

void test_1() {
#if defined(BOOST_ASIO_HAS_SERIAL_PORT)
  try {
    boost::asio::io_service ios;
    boost::asio::serial_port port(ios);

    boost::asio::serial_port_base::baud_rate baud_rate1(1000);
    port.set_option(baud_rate1);
    boost::asio::serial_port_base::baud_rate baud_rate2;
    port.get_option(baud_rate2);
    (void)static_cast<unsigned int>(baud_rate2.value());

    boost::asio::serial_port_base::flow_control flow_control1(
      boost::asio::serial_port_base::flow_control::none);
    port.set_option(flow_control1);
    boost::asio::serial_port_base::flow_control flow_control2;
    port.get_option(flow_control2);
    (void)static_cast<boost::asio::serial_port_base::flow_control::type>(
      flow_control2.value());

    boost::asio::serial_port_base::parity parity1(
      boost::asio::serial_port_base::parity::none);
    port.set_option(parity1);
    boost::asio::serial_port_base::parity parity2;
    port.get_option(parity2);
    (void)static_cast<boost::asio::serial_port_base::parity::type>(
      parity2.value());

    boost::asio::serial_port_base::stop_bits stop_bits1(
      boost::asio::serial_port_base::stop_bits::one);
    port.set_option(stop_bits1);
    boost::asio::serial_port_base::stop_bits stop_bits2;
    port.get_option(stop_bits2);
    (void)static_cast<boost::asio::serial_port_base::stop_bits::type>(
      stop_bits2.value());

    boost::asio::serial_port_base::character_size character_size1(8);
    port.set_option(character_size1);
    boost::asio::serial_port_base::character_size character_size2;
    port.get_option(character_size2);
    (void)static_cast<unsigned int>(character_size2.value());
  } catch (...) {}
#endif
}

auto main() -> decltype(0) {
  test_1();
}
