#include <boost/asio/io_service.hpp>

#include <boost/asio/detail/thread.hpp>
#include <boost/asio/deadline_timer.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>

#include <cassert>
#include <sstream>

namespace runtime {

void increment(int* count) {
  ++(*count);
}

void reset_count(boost::asio::io_service* ios, int* count) {
  if (*count > 0) {
    --(*count);
    int tmp = *count;
    ios->post(boost::bind(reset_count, ios, count));
    assert(*count == tmp);
  }
}

void reset_count_to_zero(boost::asio::io_service* ios, int* count) {
  if (*count > 0) {
    --(*count);
    ios->dispatch(boost::bind(reset_count_to_zero, ios, count));
    assert(*count == 0);
  }
}

void sleep_increment(boost::asio::io_service* ios, int* count) {
  boost::asio::deadline_timer t(*ios, boost::posix_time::seconds(2));
  t.wait();
  if (++(*count) < 3)
    ios->post(boost::bind(sleep_increment, ios, count));
}

void do_sleep_increment(boost::asio::io_service* ios, int* count) {
  boost::asio::deadline_timer t(*ios, boost::posix_time::seconds(2));
  t.wait();
  ios->post(boost::bind(sleep_increment, ios, count));
}

void throw_exception() {
  throw 1;
}

void run_ios(boost::asio::io_service* ios) {
  ios->run();
}

void stop_ios(boost::asio::io_service* ios) {
  ios->stop();
}

void reset_ios(boost::asio::io_service* ios) {
  ios->reset();
}

void test_1() {
  boost::asio::io_service ios;
  int count = 0;
  ios.post(boost::bind(increment, &count));

  assert(!ios.stopped());
  assert(count == 0);

  run_ios(&ios);
  assert(ios.stopped());
  assert(count == 1);

  reset_ios(&ios);
  assert(!ios.stopped());
  assert(count == 1);

  count = 0;
  assert(count == 0);

  for (int i = 0; i < 5; ++i) 
    ios.post(boost::bind(increment, &count));
  assert(count == 0);

  run_ios(&ios);
  assert(ios.stopped());
  assert(count == 5);

  count = 0;
  reset_ios(&ios);
  assert(count == 0);
  assert(!ios.stopped());

  boost::asio::io_service::work* w = new boost::asio::io_service::work(ios);
  ios.post(boost::bind(stop_ios, &ios));
  assert(!ios.stopped());

  run_ios(&ios);
  assert(ios.stopped());
  assert(count == 0);

  reset_ios(&ios);
  ios.post(boost::bind(increment, &count));
  delete w;

  assert(!ios.stopped());
  assert(count == 0);

  run_ios(&ios);
  stop_ios(&ios);
  assert(ios.stopped());
  assert(count == 1);

  count = 10;
  reset_ios(&ios);
  ios.post(boost::bind(reset_count, &ios, &count));
  assert(!ios.stopped());
  assert(count == 10);
  run_ios(&ios);
  assert(count == 0);

  count = 10;
  reset_ios(&ios);
  assert(!ios.stopped());
  ios.post(boost::bind(reset_count_to_zero, &ios, &count));
  assert(count == 10);
  run_ios(&ios);
  assert(ios.stopped());
  assert(count == 0);

  count = 10;
  reset_ios(&ios);
  ios.dispatch(boost::bind(reset_count_to_zero, &ios, &count));
  assert(!ios.stopped());
  assert(count == 10);
  run_ios(&ios);
  assert(ios.stopped());
  assert(count == 0);

  count = 0;
  int count2 = 0;
  reset_ios(&ios);
  assert(!ios.stopped());

  ios.post(boost::bind(do_sleep_increment, &ios, &count));
  ios.post(boost::bind(do_sleep_increment, &ios, &count2));
  boost::thread t1(boost::bind(run_ios, &ios));
  boost::thread t2(boost::bind(run_ios, &ios));
  t1.join();
  t2.join();
  assert(ios.stopped());
  assert(count == 3);
  assert(count2 == 3);

  count = 10;
  boost::asio::io_service ios2;
  ios.dispatch(ios2.wrap(boost::bind(reset_count, &ios2, &count)));
  reset_ios(&ios);
  assert(!ios.stopped());
  run_ios(&ios);
  assert(ios.stopped());
  assert(count == 10);

  run_ios(&ios2);
  assert(count == 0);

  count = 0;
  int exception_count = 0;
  reset_ios(&ios);
  assert(!ios.stopped());
  assert(count == 0);
  assert(exception_count == 0);

  ios.post(&throw_exception);
  ios.post(boost::bind(increment, &count));
  ios.post(boost::bind(increment, &count));
  ios.post(&throw_exception);
  ios.post(boost::bind(increment, &count));
  assert(!ios.stopped());
  assert(count == 0);
  assert(exception_count == 0);

  for (;;)
    try {
      run_ios(&ios);
      break;
    } catch (int) {
      ++exception_count;
    }

  assert(ios.stopped());
  assert(count == 3);
  assert(exception_count == 2);

  reset_ios(&ios);
  assert(!ios.stopped());
}

class test_service : public boost::asio::io_service::service {
public:
  static boost::asio::io_service::id id;
  test_service(boost::asio::io_service& ios)
    : boost::asio::io_service::service(ios) {}
private:
  virtual void shutdown_service() {}
};

boost::asio::io_service::id test_service::id;

void test_2() {
  boost::asio::io_service ios1;
  boost::asio::io_service ios2;
  boost::asio::io_service ios3;

  boost::asio::use_service<test_service>(ios1);
  assert(boost::asio::has_service<test_service>(ios1));
  assert(!boost::asio::has_service<test_service>(ios2));
  assert(!boost::asio::has_service<test_service>(ios3));

  test_service* s2 = new test_service(ios2);
  boost::asio::add_service(ios2, s2);
  assert(boost::asio::has_service<test_service>(ios2));
  assert(&boost::asio::use_service<test_service>(ios2) == s2);
  assert(!ios2.stopped());
  run_ios(&ios2);
  assert(ios2.stopped());
  reset_ios(&ios2);
  assert(!ios2.stopped());

  test_service* s3 = new test_service(ios2);
  try {
    boost::asio::add_service(ios2, s3);
  } catch (boost::asio::service_already_exists&) {}
  delete s3;

  test_service* s4 = new test_service(ios2);
  try {
    boost::asio::add_service(ios3, s4);
  } catch (boost::asio::invalid_service_owner&) {}
  delete s4;

  assert(!boost::asio::has_service<test_service>(ios3));
}
} // namespace

auto main() -> decltype(0) {
  runtime::test_1();
  runtime::test_2();
  return 0;
}
