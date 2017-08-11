#include <boost/asio/deadline_timer.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/placeholders.hpp>
#include <boost/asio/detail/thread.hpp>
#include <boost/bind.hpp>

#include <cassert>

void increment(int* count) {
  ++(*count);
}

void decrement_to_zero(boost::asio::deadline_timer* t, int* count) {
  if (*count > 0) {
    --(*count);

    int before_value = *count;

    t->expires_at(t->expires_at() + boost::posix_time::seconds(1));
    t->async_wait(boost::bind(decrement_to_zero, t, count));

    assert(*count == before_value);
  }
}

void increment_if_not_cancelled(int* count,
  const boost::system::error_code& ec) {
  if (!ec)
    ++(*count);
}
void cancel_timer(boost::asio::deadline_timer* t) {
  std::size_t num_cancelled = t->cancel();
  assert(num_cancelled == 1);
}

void cancle_one_timer(boost::asio::deadline_timer* t) {
  std::size_t num_cancelled = t->cancel_one();
  assert(num_cancelled == 1);
}

boost::posix_time::ptime now() {
#if defined(BOOST_DATE_TIME_HAS_HIGH_PRECISION_CLOCK)
  return boost::posix_time::microsec_clock::universal_time();
#else
  return boost::posix_time::second_clock::universal_time();
#endif
}

void deadline_timer_test() {
  boost::asio::io_service ios;
  int count = 0;

  boost::posix_time::ptime start = now();

  boost::asio::deadline_timer t1(ios, boost::posix_time::seconds(1));

  // The timer must block until after its expiry time.
  boost::posix_time::ptime end = now();
  boost::posix_time::ptime expected_end =
    end + boost::posix_time::seconds(1);
  assert(expected_end < end || expected_end == end);

  start = now();

  boost::asio::deadline_timer t2(ios, boost::posix_time::seconds(1) +
    boost::posix_time::microseconds(500000));
  t2.wait();

  // The timer must block until after its expiry time
  end = now();
  expected_end = start + boost::posix_time::seconds(1) +
    boost::posix_time::microseconds(500000);
  assert(expected_end < end || expected_end == end);

  t2.expires_at(t2.expires_at() + boost::posix_time::seconds(1));
  t2.wait();

  // The timer must block until after its expiry time
  end = now();
  expected_end += boost::posix_time::seconds(1);
  assert(expected_end < end || expected_end == end);

  start = now();

  t2.expires_from_now(boost::posix_time::seconds(1) +
    boost::posix_time::microseconds(200000));
  t2.wait();

  // The timer must block until after its expiry time
  end = now();
  expected_end = start + boost::posix_time::seconds(1) +
    boost::posix_time::microseconds(200000);
  assert(expected_end < end || expected_end == end);

  start = now();

  boost::asio::deadline_timer t3(ios, boost::posix_time::seconds(5));
  t3.async_wait(boost::bind(increment, &count));

  // No completions can be delivered until run() is called
  assert(count == 0);

  ios.run();

  // The run() call will not return until all operations have finished, and
  // this should not be until after the timer's expiry time.
  assert(count == 1);

  end = now();
  expected_end = start + boost::posix_time::seconds(1);
  assert(expected_end < end || expected_end == end);

  count = 3;
  start = now();

  boost::asio::deadline_timer t4(ios, boost::posix_time::seconds(1));
  t4.async_wait(boost::bind(decrement_to_zero, &t4, &count));

  // No completions can be delivered until run() is called.
  assert(count == 3);

  ios.reset();
  ios.run();

  // The run() call will not return until all operations have finished, and
  // this should not be until after the timer's final expiry time.
  assert(count == 0);

  end = now();
  expected_end = start + boost::posix_time::seconds(3);
  assert(expected_end < end || expected_end == end);

  count = 0;
  start = now();

  boost::asio::deadline_timer t5(ios, boost::posix_time::seconds(10));
  t5.async_wait(boost::bind(increment_if_not_cancelled, &count,
    boost::asio::placeholders::error));
  boost::asio::deadline_timer t6(ios, boost::posix_time::seconds(1));
  t6.async_wait(boost::bind(cancel_timer, &t5));

  // No completions can be delivered until run() is called
  assert(count == 0);

  ios.reset();
  ios.run();

  // The timer should have been cancelled, so count should not have changed.
  // The total run time should not have been much more than 1 second (and
  // certainly far less than 10 seconds).
  assert(count == 0);
  end = now();
  expected_end = start + boost::posix_time::seconds(2);
  assert(end < expected_end);
}
auto main() -> decltype(0) {
  return 0;
}
