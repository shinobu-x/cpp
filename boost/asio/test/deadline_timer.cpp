#include <boost/asio/deadline_timer.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/placeholders.hpp>
#include <boost/asio/detail/thread.hpp>
#include <boost/bind.hpp>

#include <cassert>

#include "async_result.hpp"

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

void cancel_one_timer(boost::asio::deadline_timer* t) {
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
  t1.wait();

  // The timer must block until after its expiry time.
  boost::posix_time::ptime end = now();
  boost::posix_time::ptime expected_end =
    start + boost::posix_time::seconds(1);
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

  // Wait on the timer again without cancelling it. This time the asynchronous
  // wait should run to completion and increment the counter.
  t5.async_wait(boost::bind(increment_if_not_cancelled, &count,
    boost::asio::placeholders::error));

  ios.reset();
  ios.run();

  // The timer should not have been cancelled, so count should have changed.
  // The total time since the timer was created should be more than 10 seconds.
  assert(count == 1);
  end = now();
  expected_end = start + boost::posix_time::seconds(10);
  assert(expected_end < end || expected_end == end);

  count = 0;
  start = now();

  // Start two waits on a timer, one of which will be cancelled. The one which
  // is not cancelled should still run to completion and increment the counter.
  boost::asio::deadline_timer t7(ios, boost::posix_time::seconds(3));
  t7.async_wait(boost::bind(increment_if_not_cancelled, &count,
    boost::asio::placeholders::error));
  t7.async_wait(boost::bind(increment_if_not_cancelled, &count,
    boost::asio::placeholders::error));
  boost::asio::deadline_timer t8(ios, boost::posix_time::seconds(1));
  t8.async_wait(boost::bind(cancel_one_timer, &t7));

  ios.reset();
  ios.run();

  // One of the waits should not have been cancelled, so count have changed.
  // The total time since the timer was created should be more than 3 seconds.
  assert(count == 1);
  end = now();
  expected_end = start + boost::posix_time::seconds(3);
  assert(expected_end < end || expected_end == end);
}

void timer_handler(const boost::system::error_code&) {}

void deadline_timer_cancel_test() {
  static boost::asio::io_service ios;

  struct timer {
    boost::asio::deadline_timer t;
    timer() : t(ios) {
      t.expires_at(boost::posix_time::pos_infin);
    }
  } timers[50];

  timers[2].t.async_wait(&timer_handler);
  timers[41].t.async_wait(&timer_handler);
  for (int i = 10; i < 20; ++i)
    timers[i].t.async_wait(&timer_handler);

  assert(timers[2].t.cancel() == 1);
  assert(timers[41].t.cancel() == 1);
  for (int i = 10; i < 20; ++i)
    assert(timers[i].t.cancel() == 1);
}

struct custom_allocation_timer_handler {
  custom_allocation_timer_handler(int* count) : count_(count) {}
  void operator()(const boost::system::error_code&) {}
  int* count_;
};

void* asio_handler_allocate(std::size_t size,
  custom_allocation_timer_handler* handler) {
  ++(*handler->count_);
  return ::operator new(size);
}

void asio_handler_deallocate(void* pointer, std::size_t,
  custom_allocation_timer_handler* handler) {
  --(*handler->count_);
  ::operator delete(pointer);
}
void deadline_timer_custom_allocation_test() {
  static boost::asio::io_service ios;
  struct timer {
    boost::asio::deadline_timer t;
    timer() : t(ios) {}
  } timers[100];

  int allocation_count = 0;

  for (int i = 0; i < 50; ++i) {
    timers[i].t.expires_at(boost::posix_time::pos_infin);
    timers[i].t.async_wait(custom_allocation_timer_handler(&allocation_count));
  }

  for (int i = 50; i < 100; ++i) {
    timers[i].t.expires_at(boost::posix_time::neg_infin);
    timers[i].t.async_wait(custom_allocation_timer_handler(&allocation_count));
  }

  for (int i = 0; i < 50; ++i)
    timers[i].t.cancel();

  ios.run();
// #include <iostream>
// std::cout << allocation_count << '\n';
  assert(allocation_count == 0);
}

void io_service_run(boost::asio::io_service* ios) {
  ios->run();
}

void deadline_timer_thread_test() {
  boost::asio::io_service ios;
  boost::asio::io_service::work w(ios);
  boost::asio::deadline_timer t1(ios);
  boost::asio::deadline_timer t2(ios);
  int count = 0;

  boost::asio::detail::thread t(boost::bind(io_service_run, &ios));

  t2.expires_from_now(boost::posix_time::seconds(2));
  t2.wait();

  t1.expires_from_now(boost::posix_time::seconds(2));
  t1.wait();

  t2.expires_from_now(boost::posix_time::seconds(4));
  t2.wait();

   ios.stop();
   t.join();
}

void deadline_timer_async_result_test() {
  boost::asio::io_service ios;
  boost::asio::deadline_timer t1(ios);

  t1.expires_from_now(boost::posix_time::seconds(1));
  // int i = t1.async_wait(archetypes::lazy_handler());
  // assert(i == 42);

  ios.run();
}

void test_1() {
  deadline_timer_test();
}

void test_2() {
  deadline_timer_cancel_test();
}

void test_3() {
  deadline_timer_custom_allocation_test();
}

void test_4() {
  deadline_timer_thread_test();
}

auto main() -> decltype(0) {
  test_1(); test_2(); test_3();
  return 0;
}
