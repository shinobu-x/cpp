#include <boost/asio/io_service.hpp>

#include <cassert>
#include <chrono>
#include <sstream>
#include <boost/asio/detail/thread.hpp>

#include <boost/asio/deadline_timer.hpp>
#include <boost/asio/steady_timer.hpp>

#include <boost/bind.hpp>


void increment(int* count) {
  ++(*count);
}

void decrement_to_zero(boost::asio::io_service* ios, int* count) {

  if (*count > 0) {
    --(*count);

    int before_value = *count;
    ios->post(boost::bind(decrement_to_zero, ios, count));

    // assert(*count == before_value);
  }
}

void nested_decrement_to_zero(boost::asio::io_service* ios, int* count) {

  if (*count > 0) {
    --(*count);

    ios->dispatch(boost::bind(nested_decrement_to_zero, ios, count));

    // assert(*count == 0);
  }
}

void sleep_increment(boost::asio::io_service* ios, int* count) {

  boost::asio::steady_timer t(*ios, std::chrono::seconds(2));
  t.wait();

  if (++(*count) < 3)
    ios->post(boost::bind(sleep_increment, ios, count));
}

void start_sleep_increments(boost::asio::io_service* ios, int* count) {

  boost::asio::steady_timer t(*ios, std::chrono::seconds(2));
  t.wait();

  ios->post(boost::bind(sleep_increment, ios, count));
}

void throw_exception() { throw 1; }

void ios_run(boost::asio::io_service* ios) { ios->run(); }

void do_test() {

  boost::asio::io_service ios;
  int count = 0;

  assert(!ios.stopped() && "Must be stopped");
  assert(count == 0 && "Must be 0");

// ******

  ios.run();

  assert(ios.stopped() && "Must run now");

// ******

  ios.reset();
  ios.post(boost::bind(increment, &count));
  ios.post(boost::bind(increment, &count));
  ios.post(boost::bind(increment, &count));
  ios.post(boost::bind(increment, &count));
  ios.post(boost::bind(increment, &count));
  ios.run();

  assert(count == 5 && "Must be 5");

  count = 0;
  ios.reset();

  boost::asio::io_service::work* w = new boost::asio::io_service::work(ios);

}
auto main() -> decltype(0)
{
  do_test();
  return 0;
}
