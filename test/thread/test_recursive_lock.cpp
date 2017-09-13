#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/lock_types.hpp>
#include <boost/thread/thread_only.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/thread/thread_time.hpp>
#include <boost/thread/condition.hpp>

#include <cassert>

template <typename mutex_type>
struct test_lock {
  void operator()() {
    mutex_type mutex;
    boost::condition condition;

    {
      typename mutex_type::scoped_lock lock(mutex, boost::defer_lock);
      assert(!lock.owns_lock());
    }

    typename mutex_type::scoped_lock lock(mutex);
    assert(lock.owns_lock());

    condition.timed_wait(lock, boost::posix_time::seconds(2));
    assert(lock.owns_lock());

    lock.unlock();
    assert(!lock.owns_lock());
    lock.lock();
    assert(lock.owns_lock());
  }
};

template <typename mutex_type>
struct test_recursive_lock {
  void operator()() {
    mutex_type mutex;
    typename mutex_type::scoped_lock lock1(mutex);
    typename mutex_type::scoped_lock lock2(mutex);
  }
};

void do_test_recursive_lock() {
  test_lock<boost::recursive_mutex>()();
  test_recursive_lock<boost::recursive_mutex>()();
}

auto main() -> decltype(0) {
  do_test_recursive_lock();
  return 0;
}
