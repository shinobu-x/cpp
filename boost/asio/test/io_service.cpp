#include <boost/asio/io_service.hpp>

#include <cassert>
#include <chrono>
#include <sstream>
#include <utility>

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
  }
}

void nested_decrement_to_zero(boost::asio::io_service* ios, int* count) {

  if (*count > 0) {
    --(*count);

    ios->dispatch(boost::bind(nested_decrement_to_zero, ios, count));
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

  assert(!ios.stopped());
  assert(count == 0);

// ******

  ios.run();
  assert(ios.stopped());

// ******

  ios.reset();
  ios.post(boost::bind(increment, &count));
  ios.post(boost::bind(increment, &count));
  ios.post(boost::bind(increment, &count));
  ios.post(boost::bind(increment, &count));
  ios.post(boost::bind(increment, &count));
  ios.run();

  assert(count == 5);

  count = 0;
  ios.reset();
  boost::asio::io_service::work* w = new boost::asio::io_service::work(ios);
  ios.post(boost::bind(&boost::asio::io_service::stop, &ios));
  ios.run();
  assert(ios.stopped());
  assert(count == 0);
  ios.reset();
  ios.post(boost::bind(increment, &count));
  delete w;
  assert(!ios.stopped());
  assert(count == 0);

  count = 10;
  ios.reset();
  ios.post(boost::bind(decrement_to_zero, &ios, &count));
  assert(!ios.stopped());
  assert(count == 10);
  ios.run();
  assert(count == 0);
  assert(ios.stopped());

  count = 10;
  ios.reset();
  ios.post(boost::bind(nested_decrement_to_zero, &ios, &count));
  ios.run();
  assert(count == 0);
  assert(ios.stopped());

  count = 10;
  ios.reset();
  ios.dispatch(boost::bind(nested_decrement_to_zero, &ios, &count));
  ios.run();
  assert(ios.stopped());
  assert(count == 0);

  int count2 = count;
  ios.reset();
  ios.post(boost::bind(start_sleep_increments, &ios, &count));
  ios.post(boost::bind(start_sleep_increments, &ios, &count2));
  boost::asio::detail::thread thread1(boost::bind(ios_run, &ios));
  boost::asio::detail::thread thread2(boost::bind(ios_run, &ios));
  thread1.join();
  thread2.join();
  assert(ios.stopped());
  assert(count == 3);
  assert(count2 == 3);

  count = 10;
  boost::asio::io_service ios2;
  ios.dispatch(ios2.wrap(boost::bind(decrement_to_zero, &ios2, &count)));
  ios.reset();
  ios.run();
  assert(count == 10);
  ios2.run();
  assert(count == 0);
}
auto main() -> decltype(0)
{
  do_test();
  return 0;
}
