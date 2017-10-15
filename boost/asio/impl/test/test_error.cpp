#include <boost/asio/error.hpp>

#include <cassert>
#include <sstream>

void test_error_code(const boost::system::error_code& ec) {
  boost::system::error_code e1(ec);
  assert(ec == e1);
  assert(!ec || e1);
  assert(!ec || !!e1);

  boost::system::error_code e2(ec);
  assert(ec == e2);
  assert(!(ec != e2));

  boost::system::error_code e3;
  e3 = ec;
  assert(ec == e3);
  assert(!(ec != e3));

  std::ostringstream os;
  os << ec;
  assert(!os.str().empty());
}

void test_1() {
  test_error_code(boost::asio::error::access_denied);
  test_error_code(boost::asio::error::address_family_not_supported);
  test_error_code(boost::asio::error::address_in_use);
  test_error_code(boost::asio::error::already_connected);
  test_error_code(boost::asio::error::already_started);
  test_error_code(boost::asio::error::connection_aborted);
  test_error_code(boost::asio::error::connection_refused);
  test_error_code(boost::asio::error::bad_descriptor);
  test_error_code(boost::asio::error::eof);
  test_error_code(boost::asio::error::fault);
  test_error_code(boost::asio::error::host_not_found);
  test_error_code(boost::asio::error::host_not_found_try_again);
  test_error_code(boost::asio::error::host_unreachable);
  test_error_code(boost::asio::error::in_progress);
  test_error_code(boost::asio::error::interrupted);
  test_error_code(boost::asio::error::invalid_argument);
  test_error_code(boost::asio::error::message_size);
  test_error_code(boost::asio::error::network_down);
  test_error_code(boost::asio::error::network_unreachable);
  test_error_code(boost::asio::error::no_descriptors);
  test_error_code(boost::asio::error::no_buffer_space);
  test_error_code(boost::asio::error::no_data);
  test_error_code(boost::asio::error::no_memory);
  test_error_code(boost::asio::error::no_permission);
  test_error_code(boost::asio::error::no_protocol_option);
  test_error_code(boost::asio::error::no_recovery);
  test_error_code(boost::asio::error::not_connected);
  test_error_code(boost::asio::error::not_socket);
  test_error_code(boost::asio::error::operation_aborted);
  test_error_code(boost::asio::error::operation_not_supported);
  test_error_code(boost::asio::error::service_not_found);
  test_error_code(boost::asio::error::shut_down);
  test_error_code(boost::asio::error::timed_out);
  test_error_code(boost::asio::error::try_again);
  test_error_code(boost::asio::error::would_block);
}

auto main() -> decltype(0) {
  test_1();
  return 0;
}
