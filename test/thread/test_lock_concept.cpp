#include <boost/mpl/vector.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/lock_types.hpp>
// #include <boost/thread/shared_mutex.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/thread/condition_variable.hpp>

#include <cassert>

#include "thread_only.hpp"
#include "../mutex/hpp/shared_mutex.hpp"
#include "../macro/config.hpp"

// Test initially locked
template <typename mutex_t, typename lock_t>
struct test_1 {
  void operator()() const {
    mutex_t m;
    lock_t l(m);

    assert(l);
    assert(l.owns_lock());
  }
};

// Test initially unlocked if other thread has lock
template <typename mutex_t, typename lock_t>
struct test_2 {
  mutex_t m;
  boost::mutex done_mutex;
  bool done;
  bool locked;
  boost::condition_variable cond;

  test_2() : done(false), locked(false) {}
  test_2(const test_2&) = delete;
  test_2(test_2&&) = delete;
  test_2& operator=(const test_2&) = delete;
  test_2& operator=(test_2&&) = delete;


  void locking_thread() {
    lock_t l1(m);
    boost::lock_guard<boost::mutex> l2(done_mutex);
    locked = l1.owns_lock();
    done = true;
    cond.notify_one();
  }

  bool is_done() const {
    return done;
  }

  void operator()() {
    lock_t l1(m);

    typedef test_2<mutex_t, lock_t> this_type;
    boost::thread t(&this_type::locking_thread, this);

    try {
      {
        boost::unique_lock<boost::mutex> l2(done_mutex);
        assert(cond.timed_wait(l2, boost::posix_time::seconds(2),
          boost::bind(&this_type::is_done, this)));
        assert(!locked);
      }

      l1.unlock();
      t.join();
    } catch (...) {
      l1.unlock();
      t.join();
      LOG;
    }
  }
};

// Test initially unlocked with try lock if other thread has unique lock
template <typename mutex_t, typename lock_t>
struct test_3 {
  mutex_t m;
  boost::mutex done_mutex;
  bool done;
  bool locked;
  boost::condition_variable cond;

  test_3() : done(false), locked(false) {}
  test_3(const test_3&) = delete;
  test_3(test_3&&) = delete;
  test_3& operator=(const test_3&) = delete;
  test_3& operator=(test_3&&) = delete;

  void locking_thread() {
    lock_t l1(m, boost::try_to_lock);
    boost::lock_guard<boost::mutex> l2(done_mutex);
    locked = l1.owns_lock();
    done = true;
    cond.notify_one();
  }

  bool is_done() const {
    return done;
  }

  void operator()() {
    boost::unique_lock<mutex_t> l1(m);

    typedef test_3<mutex_t, lock_t> this_type;
    boost::thread t(&this_type::locking_thread, this);

    try {
      {
        boost::unique_lock<boost::mutex> l2(done_mutex);
        assert(
          (cond.timed_wait(l2, boost::posix_time::seconds(2),
            boost::bind(&this_type::is_done, this))));
        assert(!locked);
      }
      l1.unlock();
      t.join();
    } catch (...) {
      l1.unlock();
      t.join();
      LOG;
    }
  }
};

// Test initially locked if other thread has shared_lock
template <typename mutex_t, typename lock_t>
struct test_4 {
  mutex_t m;
  boost::mutex done_mutex;
  bool done;
  bool locked;
  boost::condition_variable cond;

  test_4() : done(false), locked(false) {}

  void locking_thread() {
    lock_t l1(m);

    boost::lock_guard<boost::mutex> l2(done_mutex);
    locked = l1.owns_lock();
    done = true;
    cond.notify_one();
  }

  bool is_done() const {
    return done;
  }

  void operator()() {
    boost::shared_lock<mutex_t> l1(m);

    typedef test_4<mutex_t, lock_t> this_type;
    boost::thread t(&this_type::locking_thread, this);

    try {
      {
        boost::unique_lock<boost::mutex> l2(done_mutex);
        assert(cond.timed_wait(l2, boost::posix_time::seconds(2),
          boost::bind(&this_type::is_done, this)));
        assert(locked);
      }
      l1.unlock();
      t.join();
    } catch (...) {
      l1.unlock();
      t.join();
      LOG;
    }
  }
};

// Test initially unlocked with defer lock parameter
template <typename mutex_t, typename lock_t>
struct test_5 {
  void operator()() const {
    mutex_t m;
    lock_t l(m, boost::defer_lock);
    assert(!l);
    assert(!l.owns_lock());
  }
};

// Test initially locked with adopt lock parameter
template <typename mutex_t, typename lock_t>
struct test_6 {
  void operator()() const {
    mutex_t m;
    m.lock();
    lock_t l(m, boost::adopt_lock);
    assert(l);
    assert(l.owns_lock());
  }
};

// Test unlocked after unlock called
template <typename mutex_t, typename lock_t>
struct test_7 {
  void operator()() const {
    mutex_t m;
    lock_t l(m);
    l.unlock();
    assert(!l);
    assert(!l.owns_lock());
  }
};

// Test locked after lock called
template <typename mutex_t, typename lock_t>
struct test_8 {
  void operator()() const {
    mutex_t m;
    lock_t l(m, boost::defer_lock);
    l.lock();
    assert(l);
    assert(l.owns_lock());
  }
};

// Test locked after try_lock called
template <typename mutex_t, typename lock_t>
struct test_9 {
  void operator()() const {
    mutex_t m;
    lock_t l(m, boost::defer_lock);
    l.try_lock();
    assert(l);
    assert(l.owns_lock());
  }
};

// Test unlocked after try_lock if other thread has lock
template <typename mutex_t, typename lock_t>
struct test_10 {
  mutex_t m;
  boost::mutex done_mutex;
  bool done;
  bool locked;
  boost::condition_variable cond;

  test_10() : done(false), locked(false) {}

  void locking_thread() {
    lock_t l1(m, boost::defer_lock);

    boost::lock_guard<boost::mutex> l2(done_mutex);
    locked = l1.owns_lock();
    done = true;
    cond.notify_one();
  }

  bool is_done() const {
    return done;
  }

  void operator()() {
    lock_t l1(m);
    typedef test_10<mutex_t, lock_t> this_type;
    boost::thread t(&this_type::locking_thread, this);

    try {
      {
        boost::unique_lock<boost::mutex> l2(done_mutex);
        assert(cond.timed_wait(l2, boost::posix_time::seconds(2),
          boost::bind(&this_type::is_done, this)));
        assert(!locked);
      }

      l1.unlock();
      t.join();
    } catch (...) {
      l1.unlock();
      t.join();
      LOG;
    }
  }
};

// Test throw if lock_called when already locked
template <typename mutex_t, typename lock_t>
struct test_11 {
  void operator()() const {
    mutex_t m;
    lock_t l(m);
    l.lock();
    assert(l);
    assert(l.owns_lock());
    try {
      l.lock();
    } catch (boost::lock_error&) {
      LOG;
    }
  }
};

// Test throw if try lock called when already locked
template <typename mutex_t, typename lock_t>
struct test_12 {
  void operator()() const {
    mutex_t m;
    lock_t l(m);
    l.unlock();
    assert(!l);
    assert(!l.owns_lock());
    try {
      l.unlock();
    } catch (boost::lock_error&) {
      LOG;
    }
  }
};

// Test default constructed has no mutex and unlocked
template <typename lock_t>
struct test_13 {
  void operator()() const {
    lock_t l;
    assert(!l.mutex());
    assert(!l.owns_lock());
  }
};

// Test locks can be swapped
template <typename mutex_t, typename lock_t>
struct test_14 {
  void operator()() const {
    mutex_t m1;
    mutex_t m2;
    mutex_t m3;

    lock_t l1(m1);
    lock_t l2(m2);
    lock_t l3(m3);

    assert(l1.mutex() == &m1);
    assert(l2.mutex() == &m2);

    l1.swap(l2);

    assert(l1.mutex() == &m2);
    assert(l2.mutex() == &m1);

    swap(l1, l2);

    assert(l1.mutex() == &m1);
    assert(l2.mutex() == &m2);

    l1.swap(l3);
    assert(l1.mutex() == &m3);
  }
};

auto main() -> decltype(0) {
  typedef shared_mutex m;
  typedef boost::shared_lock<shared_mutex> l;
  test_1<m, l>()(); test_2<m, l>()(); test_3<m, l>()(); test_4<m, l>()();
  test_5<m, l>()(); test_6<m, l>()(); test_7<m, l>()(); test_8<m, l>()();
  test_9<m, l>()(); test_10<m, l>()(); test_11<m, l>()(); test_12<m, l>()();
  test_13<l>()(); test_14<m, l>()();
}
