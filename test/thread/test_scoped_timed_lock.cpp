#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread_only.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/thread/thread_time.hpp>
#include <boost/thread/condition.hpp>

template <typename mutex_type>
struct lock_timeout_if_other_thread_has_lock {
  mutex_type mutex;
  boost::mutex done_mutex;;
  bool done;
  bool locked;
  boost::condition_variable condition;

  lock_timeout_if_other_thread_has_lock()
    : done(false), locked(false) {}

  void locking_thread() {
    boost::unique_lock<mutex_type> lock(
      mutex, boost::posix_time::milliseconds(5));

    boost::lock_guard<boost::mutex> lock_done(done_mutex);
    locked = lock.owns_lock();
    assert(locked);
    done = lock;
    condition.notify_one();
  }

  void locking_thread_through_constructor() {
    boost::unique_lock<mutex_type> lock(
      mutex, boost::posix_time::milliseconds(50));

    boost::lock_guard<boost::mutex> lock_done(done_mutex);
    locked = lock.owns_lock();
    assert(locked);
    done = true;
    condition.notify_one();
  }

  bool is_done() const {
    return done;
  }

  typedef lock_timeout_if_other_thread_has_lock<mutex_type> this_type;

  void do_lock_timeout_if_other_thread_has_lock(void (this_type::*func)()) {
    boost::unique_lock<mutex_type> lock(mutex);

    locked = false;
    done = false;

    boost::thread t(func, this);

    try {
      {
        boost::unique_lock<boost::mutex> lock_done(done_mutex);
        assert(condition.timed_wait(
          lock_done, boost::posix_time::seconds(2),
          boost::bind(&this_type::is_done, this)));
        assert(!locked);
      }
      lock.unlock();
      t.join();
    } catch (...) {
      lock.unlock();
      t.join();
      throw;
    }
  }

  void operator()() {
  }

};

template <typename mutex_type>
struct test_scoped_timed_lock {
  static bool fake_predicate() {
    return false;
  }

  void operator()() {
    lock_timeout_if_other_thread_has_lock<mutex_type>()();

    mutex_type mutex;
    boost::condition condition;

    {
      boost::system_time timeout =
        boost::get_system_time() + boost::posix_time::milliseconds(100);
      typename mutex_type::scoped_timed_lock lock(mutex, timeout);
      assert(lock);
    }
    {
      typename mutex_type::scoped_timed_lock lock(mutex, boost::defer_lock);
      assert(!lock);
    }

    typename mutex_type::scoped_timed_lock lock(mutex);
    assert(lock);

    boost::system_time timeout1 =
      boost::get_system_time() + boost::posix_time::milliseconds(100);

    condition.timed_wait(lock, timeout1, fake_predicate);
    assert(lock);

    boost::system_time now = boost::get_system_time();
    boost::posix_time::milliseconds const timeout_resolution(20);
    assert((timeout1 - timeout_resolution) < now);

    lock.unlock();
    assert(!lock);
    lock.lock();
    assert(lock);

    lock.unlock();
    assert(!lock);
    boost::system_time timeout2 = 
      boost::get_system_time() + boost::posix_time::milliseconds(100);
    lock.timed_lock(timeout2);
    assert(lock);
    lock.unlock();
    assert(!lock);

    mutex.timed_lock(boost::posix_time::milliseconds(100));
    mutex.unlock();

    lock.timed_lock(boost::posix_time::milliseconds(100));
    assert(lock);
    lock.unlock();
    assert(!lock);
  }
};

void do_test_scoped_timed_lock() {
  test_scoped_timed_lock<boost::timed_mutex>()();
}

auto main() -> decltype(0) {
  do_test_scoped_timed_lock();
  return 0;
}
