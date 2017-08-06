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
    // boost::system_time xt = delay(0, 100);
    boost::system_time const t =
      boost::get_system_time() + boost::posix_time::milliseconds(100);
    // No one is going to notify this condition variable. So we expect to time
    // out.
    // assert(!cond.timed_wait(l, xt));
    assert(!cond.timed_wait(l, t));
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
  typedef typename M::scoped_try_lock lock_type;

  void operator() () {
    mutex_type m;
    boost::condition cond;

    {
      lock_type l(m);
      assert(l ? true : false);
    }

    {
      lock_type l(m, boost::defer_lock);
      assert(!l);
    }

    lock_type l(m);
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
  typedef M mutex_type;
  typedef typename boost::unique_lock<mutex_type> lock_type;

  M m;
  boost::mutex done_m;
  boost::condition_variable cond;

  test_lock_times_out_if_other_thread_has_lock()
    : is_done_(false), is_locked_(false) {}

  void locking_thread() {
    lock_type l(m, boost::defer_lock);
    l.timed_lock(boost::posix_time::milliseconds(50));
    boost::lock_guard<boost::mutex> lg(done_m);
    is_locked_ = l.owns_lock();
    is_done_ = true;
    cond.notify_one();
  }

  void locking_thread_through_constructor() {
    lock_type l(m, boost::posix_time::milliseconds(50));
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
    lock_type l(m);
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

// Test type: Timed lock
template <typename M>
struct test_timedlock {
  typedef M mutex_type;
  typedef typename M::scoped_timed_lock lock_type;

  static bool fake_predicate() {
    return false;
  }

  void operator() () {
    test_lock_times_out_if_other_thread_has_lock<mutex_type>()();

    mutex_type m;
    boost::condition cond;

    {
      // Cnstruct and initialize an xtime for a fast time out
      boost::system_time t =
        boost::get_system_time() + boost::posix_time::milliseconds(100);
      lock_type l(m, t);
      assert(l ? true : false);
    }

    {
      lock_type l(m, boost::defer_lock);
      assert(l ? true : false);

      // Construct and initialize an xtime for as fast time out
      boost::system_time t =
        boost::get_system_time() + boost::posix_time::milliseconds(100);

      // No one is going to notify this condition variable. So we expect to time
      // out
      assert(!cond.timed_wait(l, t, fake_predicate));
      assert(l ? true : false);

      boost::system_time now = boost::get_system_time();
      boost::posix_time::milliseconds const tr(20);
      assert((t-tr) < now);

      l.unlock();
      assert(!l);
      l.lock();
      assert(l ? true : false);
      l.unlock();
      assert(!l);
      boost::system_time target =
        boost::get_system_time() + boost::posix_time::milliseconds(100);
      assert(l.timed_lock(target));
      assert(l ? true : false);
      l.unlock();
      assert(!l);

      assert(m.timed_lock(boost::posix_time::milliseconds(100)));
      m.unlock();

      assert(l.timed_lock(boost::posix_time::milliseconds(100)));
      assert(l ? true : false);
      l.unlock();
      assert(!l);
    }
  }
};

// Test type: Recursive lock
template <typename M>
struct test_recursive_lock {
  typedef M mutex_type;
  typedef typename M::scoped_lock lock_type;

  void operator() () {
    mutex_type m;
    lock_type l1(m);
    lock_type l2(m);
  }
};

void test_1() {
  test_lock<boost::mutex>()();
}

void test_2() {
  test_lock<boost::try_mutex>()();
  test_trylock<boost::try_mutex>()();
}

void test_3() {
  test_lock<boost::timed_mutex>()();
  test_trylock<boost::timed_mutex>()();
  test_timedlock<boost::timed_mutex>()();
}

auto main() -> decltype(0) {
  test_1(); test_2();
  return 0;
}
