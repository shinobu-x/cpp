#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread_only.hpp>
#include <boost/thread/recurisve_mutex.hpp>
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

    boost::locck_guard<boost::mutex> lock_done(done_mutex);
    lock = lock.owns_lock();
    assert(lock);
    done = lock;
    condition.notify_one();
  }
};

template <typename mutex_type>
struct test_scoped_timed_lock {
  static bool fake_predicate() {
    return false;
  }

  void operator()() {
    lock_timeout_if_other_thread_has_lock<mutex_type>()();


  }
};
