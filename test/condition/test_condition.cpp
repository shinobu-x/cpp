#include <boost/thread/condition.hpp>
#include <boost/thread/thread_only.hpp>

#include <cassert>

struct test_data {
  test_data() : notified_(0), wakeup_(0) {}

  boost::mutex mutex;
  boost::condition_variable condition;
  int notified_;
  int wakeup_;
};

void test_condition_thread(test_data* data) {
  boost::unique_lock<boost::mutex> lock(data->mutex);
  assert(lock.owns_lock());

  while (!(data->notified_ > 0))
    data->condition.wait(lock);

  assert(lock.owns_lock());

  data->wakeup_++;
}

struct condition_predicate {
  condition_predicate(int& var, int val) : var_(var), val_(val) {}

  bool operator()() {
    return var_ == val_;
  }

  int& var_;
  int val_;

private:
  void operator=(condition_predicate);
};

void test_condition_waits(test_data* data) {
  boost::unique_lock<boost::mutex> lock(data->mutex);

  assert(lock.owns_lock());

  {
    while (data->notified_ != 1)
      data->condition.wait(lock);

    assert(lock.owns_lock());
    assert(data->notified_ == 1);

    data->wakeup_++;
    data->condition.notify_one();
  }

  {
    data->condition.wait(lock, condition_predicate(data->notified_, 2));
    assert(lock.owns_lock());
    assert(data->notified_ == 2);
    data->wakeup_++;
    data->condition.notify_one();
  }

  {
    while (data->notified_ != 3)
      data->condition.timed_wait(lock, boost::posix_time::seconds(1));

    assert(lock.owns_lock());
    assert(data->notified_ == 3);

    data->wakeup_++;
    data->condition.notify_one();
  }

  {
    condition_predicate predicate(data->notified_, 4);
    assert(data->condition.timed_wait(
      lock, boost::posix_time::seconds(2), predicate));
    assert(lock.owns_lock());
    assert(predicate());
    assert(data->notified_ == 4);
    data->wakeup_++;
    data->condition.notify_one();
  }

  {
    condition_predicate predicate(data->notified_, 5);
    assert(data->condition.timed_wait(
      lock, boost::posix_time::seconds(10), predicate));
    assert(lock.owns_lock());
    assert(predicate());
    assert(data->notified_ == 5);
    data->wakeup_++;
    data->condition.notify_one();
  }

};

void do_test_condition_waits() {
  test_data data;
  boost::thread thread(boost::bind(&test_condition_waits, &data));

  {
    boost::unique_lock<boost::mutex> lock(data.mutex);
    assert(lock.owns_lock());

    boost::system_time const start = boost::get_system_time();
    boost::system_time const timeout =
      start + boost::posix_time::seconds(5);
    boost::thread::sleep(timeout);
    data.notified_++;
    data.condition.notify_one();
    while (data.wakeup_ != 1)
      data.condition.wait(lock);
    assert(lock.owns_lock());
    assert(data.wakeup_ == 1);

    boost::thread::sleep(timeout);
    data.notified_++;
    data.condition.notify_one();
    while (data.wakeup_ != 2)
      data.condition.wait(lock);
    assert(lock.owns_lock());
    assert(data.wakeup_ == 2);

    boost::thread::sleep(timeout);
    data.notified_++;
    data.condition.notify_one();
    while (data.wakeup_ != 3)
      data.condition.wait(lock);
    assert(lock.owns_lock());
    assert(data.wakeup_ == 3);

    boost::thread::sleep(timeout);
    data.notified_++;
    data.condition.notify_one();
    while (data.wakeup_ != 4)
      data.condition.wait(lock);
    assert(lock.owns_lock());

    boost::thread::sleep(timeout);
    data.notified_++;
    data.condition.notify_one();
    while (data.wakeup_ != 5)
      data.condition.wait(lock);
    assert(lock.owns_lock());
    assert(data.wakeup_ == 5);
  }

  thread.join();
  assert(data.wakeup_ == 5);
}

void do_test_condition_wait_is_an_interruption_point() {
  test_data data;
  boost::thread thread(boost::bind(&test_condition_thread, &data));

  thread.interrupt();
  thread.join();
  assert(data.wakeup_ == 0);
}

auto main() -> decltype(0) {
  do_test_condition_waits(); do_test_condition_wait_is_an_interruption_point();
  return 0;
}
