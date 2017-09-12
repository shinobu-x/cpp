#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/lock_types.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/thread/thread_time.hpp>
#include <boost/thread/condition.hpp>

#include <cassert>

template <typename mutex_type>
struct test_scoped_try_lock {
  void operator()() {
    mutex_type mutex;
    boost::condition condition;

    {
      typename mutex_type::scoped_try_lock lock(mutex);
      assert(lock);
    }

    {
      typename mutex_type::scoped_try_lock lock(mutex, boost::defer_lock);
      assert(!lock);
    }

    typename mutex_type::scoped_try_lock lock(mutex);
    assert(lock);

    assert(!condition.timed_wait(lock, boost::posix_time::seconds(2)));
    assert(lock);

    try {
      for (unsigned i = 0; i < 10; ++i) {
        lock.unlock();
        assert(!lock);
        lock.lock();
        assert(lock);
      }
    } catch (...) {}
  }
};

void do_test_scoped_try_lock() {
  boost::thread_group threads;
  try {
    for (unsigned i = 0; i < 10; ++i)
      threads.create_thread(test_scoped_try_lock<boost::mutex>());

    threads.join_all();
  } catch (...) {
    threads.interrupt_all();
    threads.join_all();
    throw;
  }
}

auto main() -> decltype(0) {
  do_test_scoped_try_lock();
  return 0;
}
