#include <boost/thread/mutex.hpp>
#include <boost/thread/lock_types.hpp>
// #include <boost/thread/thread_only.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/thread/thread_time.hpp>
#include <boost/thread/condition.hpp>

#include "thread_only.hpp"
#include "../utils/utils.hpp"

#include <cassert>

// Test type: Lock
template <typename M>
struct test_lock {
  typedef M mutex_type;
  typedef typename M::scoped_lock lock_type;

  void operator() () {
    mutex_type m;
    boost::condition cond;

    {
      lock_type l(m, boost::defer_lock);
      assert(!l);
    }

    lock_type l(m);
    assert(l ? true : false);

    // Construct and initialize an xtime for a fasrt time out.
    boost::xtime xt = delay(0, 100);

    // No one is going to notify this condition variable. So we expect to time
    // out.
    assert(!cond.timed_wait(l, xt));
    assert(l ? true : false);

    l.unlock();
    assert(!l);
    l.lock();
    assert(l ? true : false);
  }
};

// Test type: Trylock
template <typename M>
struct test_trylock {
  typedef M mutex_type;
  typedef typename M::scoped_try_lock try_lock_type;

  void operator() () {
    mutex_type m;
    boost::condition cond;

    {
      try_lock_type l(m);
      assert(l ? true : false);
    }

    {
      try_lock_type l(m, boost::defer_lock);
      assert(!l);
    }

    try_lock_type l(m);
    assert(l ? true : false);

    // Construct and initialize an extime for a fast time out.
    boost::xtime xt = delay(0, 100);

    // No one is going to notify this condition variable. So we expect to time
    // out.
    assert(!cond.timed_wait(l, xt));
    assert(l ? true : false);

    l.unlock();
    assert(!l);
    l.lock();
    assert(l ? true : false);
    l.unlock();
    assert(!l);
    assert(l.try_lock());
    assert(l ? true : false);
  }
};

// Test type: Lock time out
template <typename M>
struct test_lock_times_out_if_other_thread_has_lock {
private:
  bool is_done_;
  bool is_locked_;

public:
  typedef boost::unique_lock<M> lock;

  M m;
  boost::mutex done_m;
  boost::condition_variable cond;

  test_lock_times_out_if_other_thread_has_lock()
    : is_done_(false), is_locked_(false) {}

  void locking_thread() {
    lock l(m, boost::defer_lock);
    l.timed_lock(boost::posix_time::milliseconds(50));
    boost::lock_guard<boost::mutex> lg(done_m);
    is_locked_ = l.owns_lock();
    is_done_ = true;
    cond.notify_one();
  }

  void locking_thread_through_constructor() {
    lock l(m, boost::posix_time::milliseconds(50));
    boost::lock_guard<boost::mutex> lg(done_m);
    is_locked_ = l.owns_lock();
    is_done_ = true;
    cond.notify_one();
  }

  bool is_done() const {
    return is_done_;
  }

  typedef test_lock_times_out_if_other_thread_has_lock<M> this_type;

  void test(void (this_type::*test_func)()) {
    lock l(m);
    is_locked_ = false;
    is_done_ = false;

    boost::thread t(test_func, this);

    try {
      {
        boost::unique_lock<boost::mutex> l(done_m);
        assert(cond.timed_wait(l, boost::posix_time::seconds(2),
          boost::bind(&this_type::is_done_, this)));
        assert(!is_locked_);
      }

      l.unlock();
      assert(!l);
      t.join();
    } catch (...) {
      l.unlock();
      assert(!l);
      t.join();
      throw;
    }
  }
  
  void operator() () {
    test(&this_type::locking_thread);
    test(&this_type::locking_thread_through_constructor);
  }
};
auto main() -> decltype(0) {
  return 0;
}
