#include <boost/asio/signal_set.hpp>
#include <boost/asio/io_service.hpp>

#include "async_result.hpp"

void signal_handler(const boost::system::error_code&, int) {}

void test_1() {
  try {
    boost::asio::io_service ios;
    lazy_handler lazy;
    boost::system::error_code ec;

    boost::asio::signal_set signal_set1(ios);
    boost::asio::signal_set signal_set2(ios, 1);
    boost::asio::signal_set signal_set3(ios, 1, 2);
    boost::asio::signal_set signal_set4(ios, 1, 2, 3);

    boost::asio::io_service& ios_ref = signal_set1.get_io_service();
    (void)ios_ref;

    signal_set1.add(1);
    signal_set1.add(1, ec);

    signal_set1.remove(1);
    signal_set1.remove(1, ec);

    signal_set1.clear();
    signal_set1.clear(ec);

    signal_set1.async_wait(&signal_handler);
    int i = signal_set1.async_wait(lazy);
    (void)i;
  } catch (...) {}
}

auto main() -> decltype(0) {
  return 0;
}
